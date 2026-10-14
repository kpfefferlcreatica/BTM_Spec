"""aFRR capacity: deterministic award against the realised DE marginal price,
headroom and SOC corridor on the awarded MW, reservation price mu, and the
fixed point between mu and the bid policy. One 4-slot block = 1 h."""

import numpy as np
import pytest

from pf_bound.bid_policy import solve_day_with_bid_policy
from pf_bound.day_model import DN, UP, AfrrCapacityDay, DayInputs, capacity_shadow_values, solve_day

TOL = 1e-6
NO_BID = np.nan


def day(da, *, soc_start, marginal=(np.inf, np.inf), history=None):
    h = len(da)
    return DayInputs(
        da_price=np.asarray(da, float),
        id_price=None,
        load_mw=np.zeros(h),
        pv_avail_mw=np.zeros(h),
        soc_start_mwh=soc_start,
        peak_in_mw=0.0,
        terminal_value_eur_per_mwh=0.0,
        afrr_capacity=AfrrCapacityDay(
            block_of_slot=np.zeros(h, int),
            marginal_price=np.array([marginal], float),
            history=np.zeros((0, 1, 2)) if history is None else np.asarray(history, float),
        ),
    )


def bids(up=NO_BID, dn=NO_BID):
    return np.array([[up, dn]], float)


def test_awarded_volume_limited_by_corridor(config_factory):
    # corridor 1 slot, eta 1: soc >= r * 0.25 -> r <= 2 at soc 0.5; price 0 so selling
    # the stored energy is worth nothing
    cfg = config_factory(eta=1.0, corridor_slots=1)
    res = solve_day(day([0] * 4, soc_start=0.5, marginal=(12, 12)), cfg, capacity_bids=bids(up=10))
    assert res.schedule["r_up_mw"].iloc[0] == pytest.approx(2.0)
    assert res.pnl["afrr_capacity"] == pytest.approx(10 * 2 * 1.0)


def test_bid_above_marginal_is_not_awarded(config_factory):
    cfg = config_factory(eta=1.0, corridor_slots=1)
    res = solve_day(day([50] * 4, soc_start=0.5, marginal=(12, 12)), cfg, capacity_bids=bids(up=15))
    assert res.schedule["r_up_mw"].sum() == pytest.approx(0.0, abs=TOL)
    assert res.pnl["afrr_capacity"] == pytest.approx(0.0, abs=TOL)


def test_headroom_trades_reserve_against_spot(config_factory):
    cfg = config_factory(eta=1.0, corridor_slots=0)
    spike = [0, 0, 0, 1000]
    cheap = solve_day(day(spike, soc_start=1.0, marginal=(2000, 2000)), cfg, capacity_bids=bids(up=1))
    assert cheap.schedule["r_up_mw"].iloc[0] == pytest.approx(0.0, abs=TOL)
    rich = solve_day(day(spike, soc_start=1.0, marginal=(2000, 2000)), cfg, capacity_bids=bids(up=1000))
    assert rich.schedule["r_up_mw"].iloc[0] == pytest.approx(4.0)
    assert rich.schedule["p_discharge_mw"].max() == pytest.approx(0.0, abs=TOL)


def test_down_reserve_respects_upper_corridor(config_factory):
    # soc <= 1 - r_dn * 0.25 at soc 0.5 -> r_dn <= 2
    cfg = config_factory(eta=1.0, corridor_slots=1)
    res = solve_day(day([50] * 4, soc_start=0.5, marginal=(12, 12)), cfg, capacity_bids=bids(dn=10))
    assert res.schedule["r_dn_mw"].iloc[0] == pytest.approx(2.0)


def test_reservation_price_is_lost_spot_value_per_mw_hour(config_factory):
    # one MW of up reserve blocks one MW of discharge in the 1000 EUR/MWh slot: 250 EUR over a 1 h block
    cfg = config_factory(eta=1.0, corridor_slots=0)
    inp = day([0, 0, 0, 1000], soc_start=1.0, marginal=(2000, 2000))
    res = solve_day(inp, cfg, capacity_bids=bids(up=1))
    mu = capacity_shadow_values(inp, cfg, res)
    assert mu[0, UP] == pytest.approx(250.0)
    assert mu[0, DN] == pytest.approx(0.0, abs=TOL)


def spike_day():
    hist = np.array([[[100.0, 0.0]], [[300.0, 0.0]], [[500.0, 0.0]]])
    return day([0, 0, 0, 1000], soc_start=1.0, marginal=(400, 0), history=hist)


def test_fixed_point_converges_on_policy_bid(config_factory):
    # mu=0 -> bid 300 (awarded, mu=250) -> bid 500 (not awarded, mu=250) -> bid 500: converged
    cfg = config_factory(eta=1.0, corridor_slots=0)
    out = solve_day_with_bid_policy(spike_day(), cfg)
    assert out.converged
    assert out.bids[0, UP] == pytest.approx(500.0)
    assert out.result.schedule["r_up_mw"].sum() == pytest.approx(0.0, abs=TOL)
    assert out.result.objective_eur == pytest.approx(1000.0)
    assert len(out.log) == 2


def test_unconverged_fixed_point_keeps_best_iterate(config_factory):
    cfg = config_factory(eta=1.0, corridor_slots=0, bid_policy_max_iter=1)
    out = solve_day_with_bid_policy(spike_day(), cfg)
    assert not out.converged
    assert out.bids[0, UP] == pytest.approx(300.0)
    assert out.result.objective_eur == pytest.approx(1200.0)


def test_award_and_reserve_are_per_block(config_factory):
    # two blocks of two slots; only block 1 awarded. POI export limit 1 MW: p_poi - r_up >= -1,
    # so charging at 1 MW (free at price 0) lifts the cap to 2 MW
    cfg = config_factory(eta=1.0, corridor_slots=0, poi_limit=1.0)
    inp = DayInputs(
        da_price=np.zeros(4),
        id_price=None,
        load_mw=np.zeros(4),
        pv_avail_mw=np.zeros(4),
        soc_start_mwh=0.5,
        peak_in_mw=0.0,
        terminal_value_eur_per_mwh=0.0,
        afrr_capacity=AfrrCapacityDay(
            block_of_slot=np.array([0, 0, 1, 1]),
            marginal_price=np.array([[5.0, 0.0], [20.0, 0.0]]),
            history=np.zeros((0, 2, 2)),
        ),
    )
    res = solve_day(inp, cfg, capacity_bids=np.array([[10.0, np.nan], [10.0, np.nan]]))
    assert res.awarded.tolist() == [[False, False], [True, False]]
    assert res.schedule["r_up_mw"].tolist() == pytest.approx([0, 0, 2, 2])
    assert res.pnl["afrr_capacity"] == pytest.approx(10 * 2 * 0.5)
