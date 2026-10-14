"""Day-to-day driver: SOC and running peak carried across days, V from D+1 DA,
capacity lookback strictly before the gate, P&L and export."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pf_bound.backtest import DayData, lookback_history, run_backtest
from pf_bound.terminal import terminal_value

TOL = 1e-6


def make_day(d, da, *, load=None, capacity_marginal=None):
    h = len(da)
    start = pd.Timestamp(d, tz="Europe/Berlin").tz_convert("UTC")
    return DayData(
        date=d,
        slot_start_utc=pd.date_range(start, periods=h, freq="15min"),
        da_price=np.asarray(da, float),
        id_price=None,
        load_mw=np.zeros(h) if load is None else np.asarray(load, float),
        pv_avail_mw=np.zeros(h),
        block_of_slot=None if capacity_marginal is None else np.zeros(h, int),
        capacity_marginal=None if capacity_marginal is None else np.asarray(capacity_marginal, float),
        energy=None,
    )


def test_soc_is_carried_from_day_to_day(config_factory):
    d0 = date(2026, 3, 2)
    days = [make_day(d0, [0, 100]), make_day(d0 + timedelta(1), [0, 100])]
    out = run_backtest(days, config_factory(), soc_start_mwh=0.3, after_last_day_da=[0, 0])
    daily = out.daily
    assert daily["soc_start_mwh"].iloc[0] == pytest.approx(0.3)
    assert daily["soc_start_mwh"].iloc[1] == pytest.approx(daily["soc_end_mwh"].iloc[0])


def test_running_peak_resets_at_month_boundary(config_factory):
    cfg = config_factory(peak_price=1000.0)
    days = [make_day(date(2026, 1, 30), [0, 0], load=[2, 2]),
            make_day(date(2026, 1, 31), [0, 0], load=[2, 2]),
            make_day(date(2026, 2, 1), [0, 0], load=[2, 2])]
    out = run_backtest(days, cfg, soc_start_mwh=0.0, after_last_day_da=[0, 0])
    assert out.daily["peak"].tolist() == pytest.approx([-2000.0, 0.0, -2000.0])
    assert out.monthly.loc["2026-01", "peak"] == pytest.approx(-2000.0)


def test_terminal_value_reads_next_day_prices(config_factory):
    cfg = config_factory()
    d0 = date(2026, 3, 2)
    days = [make_day(d0, [10, 20]), make_day(d0 + timedelta(1), [30, 70])]
    out = run_backtest(days, cfg, soc_start_mwh=0.0, after_last_day_da=[5, 90])
    assert out.daily["terminal_value_eur_per_mwh"].iloc[0] == pytest.approx(terminal_value([30, 70], cfg))
    assert out.daily["terminal_value_eur_per_mwh"].iloc[1] == pytest.approx(terminal_value([5, 90], cfg))


def test_last_day_needs_following_day_prices(config_factory):
    with pytest.raises(ValueError, match="after_last_day_da"):
        run_backtest([make_day(date(2026, 3, 2), [0, 0])], config_factory(), soc_start_mwh=0.0)


def test_days_must_be_consecutive(config_factory):
    days = [make_day(date(2026, 3, 2), [0, 0]), make_day(date(2026, 3, 4), [0, 0])]
    with pytest.raises(ValueError, match="consecutive"):
        run_backtest(days, config_factory(), soc_start_mwh=0.0, after_last_day_da=[0, 0])


def test_lookback_stops_two_days_before_delivery():
    hist = {date(2026, 3, d): np.full((1, 2), float(d)) for d in range(1, 11)}
    window = lookback_history(hist, date(2026, 3, 10), days=3)
    # gate D-1 09:00: D-1 and D are unknown, so the window is D-4 .. D-2
    assert window[:, 0, 0].tolist() == [6.0, 7.0, 8.0]


def test_cash_pnl_excludes_terminal_values(config_factory):
    d0 = date(2026, 3, 2)
    days = [make_day(d0, [0, 0]), make_day(d0 + timedelta(1), [0, 0])]
    out = run_backtest(days, config_factory(), soc_start_mwh=0.0, after_last_day_da=[100, 100])
    # day 2 charges for free to the terminal value; nothing of it is cash
    assert out.daily["terminal"].iloc[1] > 0
    assert out.cash_total_eur == pytest.approx(0.0, abs=TOL)
    assert out.final_soc_value_eur == pytest.approx(out.daily["terminal"].iloc[-1])


def test_capacity_bids_use_history_and_are_logged(config_factory):
    cfg = config_factory(eta=1.0, corridor_slots=0)
    d0 = date(2026, 3, 10)
    hist = {d0 - timedelta(k): np.array([[20.0, 20.0]]) for k in range(2, 40)}
    days = [make_day(d0, [0] * 4, capacity_marginal=[[25.0, 25.0]])]
    out = run_backtest(days, cfg, soc_start_mwh=0.5, after_last_day_da=[0] * 4, capacity_history=hist)
    bids = out.bids
    assert set(bids["direction"]) == {"up", "dn"}
    assert (bids["bid_eur_per_mw_h"] == 20.0).all()
    assert bids["awarded"].all()
    assert out.daily["afrr_capacity"].iloc[0] > 0


def test_export_writes_parquet(config_factory, tmp_path):
    d0 = date(2026, 3, 2)
    out = run_backtest([make_day(d0, [0, 100])], config_factory(), soc_start_mwh=0.0, after_last_day_da=[0, 0])
    out.to_parquet(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} >= {"daily.parquet", "schedule.parquet", "monthly.parquet"}
    assert len(pd.read_parquet(tmp_path / "schedule.parquet")) == 2
