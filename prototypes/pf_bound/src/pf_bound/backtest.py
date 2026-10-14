"""Day-to-day driver for the foresight bound.

Carries SOC and the running import peak across consecutive delivery days,
resets the peak at each accounting-period boundary (Berlin civil calendar),
values end-of-day SOC with V from the next day's realised DA prices, and builds
the capacity lookback from days strictly before the gate (D-1 09:00, so the
latest complete day is D-2).

Daily objectives include the terminal value, which steers decisions but is not
cash. `cash_total_eur` sums every term except `terminal`; the value of the final
SOC is reported separately as `final_soc_value_eur`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pf_bound.bid_policy import solve_day_with_bid_policy
from pf_bound.config import SiteConfig
from pf_bound.day_model import DN, UP, AfrrCapacityDay, AfrrEnergyDay, DayInputs, solve_day
from pf_bound.terminal import terminal_value

DIRECTIONS = {UP: "up", DN: "dn"}


@dataclass(frozen=True)
class DayData:
    date: date  # Berlin delivery day
    slot_start_utc: pd.DatetimeIndex  # [H]; H = 92 / 96 / 100 on DST days
    da_price: np.ndarray
    id_price: np.ndarray | None
    load_mw: np.ndarray
    pv_avail_mw: np.ndarray
    block_of_slot: np.ndarray | None  # int [H]; None without aFRR capacity
    capacity_marginal: np.ndarray | None  # EUR/MW/h [B, 2], realised DE marginal price
    energy: AfrrEnergyDay | None


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame  # one row per day: P&L terms, state, V, fixed-point status
    schedule: pd.DataFrame  # one row per slot
    bids: pd.DataFrame  # one row per day, block and direction
    monthly: pd.DataFrame  # P&L terms summed per Berlin month

    @property
    def cash_total_eur(self) -> float:
        return float(self.daily[_cash_terms(self.daily)].to_numpy().sum())

    @property
    def final_soc_value_eur(self) -> float:
        return float(self.daily["terminal"].iloc[-1])

    def to_parquet(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.daily.to_parquet(directory / "daily.parquet")
        self.schedule.to_parquet(directory / "schedule.parquet")
        self.monthly.to_parquet(directory / "monthly.parquet")
        if len(self.bids):
            self.bids.to_parquet(directory / "bids.parquet")


def lookback_history(history: Mapping[date, np.ndarray], delivery: date, days: int) -> np.ndarray:
    """Marginal capacity prices [N, B, 2] of the latest `days` days up to D-2, oldest first."""
    cutoff = delivery - timedelta(days=2)
    known = sorted(d for d in history if d <= cutoff)[-days:]
    if not known:
        shape = next(iter(history.values())).shape if history else (6, 2)
        return np.zeros((0, *shape))
    return np.stack([history[d] for d in known])


def run_backtest(
    days: Sequence[DayData],
    cfg: SiteConfig,
    *,
    soc_start_mwh: float,
    after_last_day_da: Sequence[float] | None = None,
    capacity_history: Mapping[date, np.ndarray] | None = None,
) -> BacktestResult:
    _check_days(days)
    if after_last_day_da is None:
        raise ValueError("after_last_day_da is required: V of the last day reads the following day's DA prices")
    history = dict(capacity_history or {})
    for day in days:
        if day.capacity_marginal is not None:
            history[day.date] = day.capacity_marginal

    soc, peak, period = soc_start_mwh, 0.0, None
    daily_rows, schedules, bid_rows = [], [], []
    for i, day in enumerate(days):
        key = _period_key(day.date, cfg.tariff.peak_accounting_period)
        if key != period:
            peak, period = 0.0, key
        next_da = days[i + 1].da_price if i + 1 < len(days) else np.asarray(after_last_day_da, float)
        v = terminal_value(next_da, cfg)

        inp = DayInputs(
            da_price=day.da_price,
            id_price=day.id_price,
            load_mw=day.load_mw,
            pv_avail_mw=day.pv_avail_mw,
            soc_start_mwh=soc,
            peak_in_mw=peak,
            terminal_value_eur_per_mwh=v,
            afrr_energy=day.energy,
        )
        converged, iterations = None, None
        if day.capacity_marginal is not None:
            inp = replace(inp, afrr_capacity=AfrrCapacityDay(
                block_of_slot=day.block_of_slot,
                marginal_price=day.capacity_marginal,
                history=lookback_history(history, day.date, cfg.afrr.capacity_lookback_days),
            ))
            outcome = solve_day_with_bid_policy(inp, cfg)
            res, converged, iterations = outcome.result, outcome.converged, len(outcome.log)
            bid_rows += _bid_rows(day, outcome)
        else:
            res = solve_day(inp, cfg)

        daily_rows.append({
            "date": day.date,
            "objective_eur": res.objective_eur,
            **res.pnl,
            "soc_start_mwh": soc,
            "soc_end_mwh": res.soc_end_mwh,
            "peak_in_mw": peak,
            "peak_out_mw": res.peak_out_mw,
            "terminal_value_eur_per_mwh": v,
            "bid_policy_converged": converged,
            "bid_policy_iterations": iterations,
        })
        schedules.append(res.schedule.assign(date=day.date).set_index(day.slot_start_utc.rename("slot_start_utc")))
        soc, peak = res.soc_end_mwh, res.peak_out_mw

    daily = pd.DataFrame(daily_rows).set_index("date")
    month = pd.PeriodIndex([pd.Period(d, "M") for d in daily.index]).astype(str)
    monthly = daily[_cash_terms(daily) + ["terminal"]].groupby(month).sum()
    monthly.index.name = "month"
    return BacktestResult(daily, pd.concat(schedules), pd.DataFrame(bid_rows), monthly)


def _cash_terms(daily: pd.DataFrame) -> list[str]:
    terms = ["da", "id", "id_transaction_cost", "network_volumetric", "levies", "peak", "afrr_capacity", "afrr_energy"]
    return [t for t in terms if t in daily.columns]


def _bid_rows(day: DayData, outcome) -> list[dict]:
    res = outcome.result
    rows = []
    for b in range(outcome.bids.shape[0]):
        for d, tag in DIRECTIONS.items():
            rows.append({
                "date": day.date,
                "block": b,
                "direction": tag,
                "bid_eur_per_mw_h": outcome.bids[b, d],
                "marginal_eur_per_mw_h": day.capacity_marginal[b, d],
                "awarded": bool(res.awarded[b, d]),
                "reserve_mw": float(res.reserve_mw[b, d]),
                "mu_eur_per_mw_h": outcome.mu[b, d],
            })
    return rows


def _period_key(d: date, period: str) -> tuple:
    return (d.year, d.month) if period == "month" else (d.year,)


def _check_days(days: Sequence[DayData]) -> None:
    if not days:
        raise ValueError("no days to run")
    for prev, cur in zip(days, days[1:]):
        if cur.date - prev.date != timedelta(days=1):
            raise ValueError(f"days must be consecutive: {prev.date} -> {cur.date}")
