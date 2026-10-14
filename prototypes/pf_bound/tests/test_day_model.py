"""Hand-checkable instances for the daily model core: SOC, DA, ID, peak, tariffs,
curtailment, cycle limit, terminal value. Slots are 15 min (0.25 h)."""

import numpy as np
import pytest

from pf_bound.day_model import DayInputs, solve_day

TOL = 1e-6


def inputs(da, *, load=None, pv=None, id_price=None, soc_start=0.0, peak_in=0.0, terminal=0.0):
    h = len(da)
    return DayInputs(
        da_price=np.asarray(da, float),
        id_price=None if id_price is None else np.asarray(id_price, float),
        load_mw=np.zeros(h) if load is None else np.asarray(load, float),
        pv_avail_mw=np.zeros(h) if pv is None else np.asarray(pv, float),
        soc_start_mwh=soc_start,
        peak_in_mw=peak_in,
        terminal_value_eur_per_mwh=terminal,
    )


def test_flat_price_leaves_battery_idle(config_factory):
    res = solve_day(inputs([50, 50, 50, 50]), config_factory())
    assert res.schedule["p_charge_mw"].sum() == pytest.approx(0, abs=TOL)
    assert res.schedule["p_discharge_mw"].sum() == pytest.approx(0, abs=TOL)
    assert res.objective_eur == pytest.approx(0, abs=TOL)


def test_two_slot_arbitrage(config_factory):
    # charge 4 MW * 0.25 h = 1 MWh AC -> 0.9 MWh stored; discharge 0.9 * 0.9 = 0.81 MWh AC
    res = solve_day(inputs([0, 100]), config_factory(eta=0.9))
    assert res.objective_eur == pytest.approx(81.0)
    assert res.pnl["da"] == pytest.approx(81.0)
    assert res.soc_end_mwh == pytest.approx(0.0, abs=TOL)


def test_soc_stays_in_band_on_random_prices(config_factory):
    rng = np.random.default_rng(7)
    cfg = config_factory(soc_min=0.1, soc_max=0.9)
    res = solve_day(inputs(rng.normal(50, 40, 24), soc_start=0.5), cfg)
    soc = res.schedule["soc_start_mwh"].to_numpy()
    assert soc.min() >= 0.1 - TOL and soc.max() <= 0.9 + TOL
    assert 0.1 - TOL <= res.soc_end_mwh <= 0.9 + TOL


def test_peak_shaving_spreads_stored_energy(config_factory):
    # 0.9 MWh AC available over 2 slots shaves 1.8 MW each: peak 2.0 -> 0.2
    cfg = config_factory(eta=0.9, peak_price=1000.0)
    res = solve_day(inputs([0, 0], load=[2, 2], soc_start=1.0), cfg)
    assert res.peak_out_mw == pytest.approx(0.2)
    assert res.pnl["peak"] == pytest.approx(-200.0)


def test_running_peak_is_free_up_to_carried_in_value(config_factory):
    cfg = config_factory(peak_price=1000.0)
    res = solve_day(inputs([0, 0], load=[1.2, 1.2], peak_in=1.5), cfg)
    assert res.pnl["peak"] == pytest.approx(0.0, abs=TOL)
    assert res.peak_out_mw == pytest.approx(1.5)


def test_import_charges_apply_to_import_only(config_factory):
    cfg = config_factory(volumetric=10.0, levies=5.0)
    res = solve_day(inputs([0, 0], load=[1, 0], pv=[0, 1]), cfg)
    assert res.pnl["network_volumetric"] == pytest.approx(-10.0 * 0.25)
    assert res.pnl["levies"] == pytest.approx(-5.0 * 0.25)


def test_negative_price_curtails_pv(config_factory):
    # simultaneous charge/discharge would burn imported energy at a profit
    res = solve_day(inputs([-50], pv=[3], soc_start=1.0), config_factory(forbid_simultaneous=True))
    assert res.schedule["pv_curtailed_mw"].iloc[0] == pytest.approx(3.0)
    assert res.objective_eur == pytest.approx(0.0, abs=TOL)


def test_cycle_limit_caps_discharged_energy(config_factory):
    prices = [0, 100, 0, 100]
    free = solve_day(inputs(prices), config_factory())
    capped = solve_day(inputs(prices), config_factory(cycle_limit=1.0))
    discharged = capped.schedule["p_discharge_mw"].sum() * 0.25
    assert discharged <= 1.0 + TOL
    assert capped.objective_eur < free.objective_eur - 1.0


def test_terminal_value_fills_battery(config_factory):
    res = solve_day(inputs([10, 10], terminal=100.0), config_factory(eta=0.9))
    assert res.soc_end_mwh == pytest.approx(1.0)
    assert res.pnl["terminal"] == pytest.approx(100.0)


def test_id_used_only_when_spread_beats_cost(config_factory):
    cfg = config_factory(id_enabled=True, id_cost=0.5, poi_limit=1.0)
    small = solve_day(inputs([50], pv=[1], id_price=[50.3], soc_start=1.0), cfg)
    assert small.schedule["x_id_mw"].abs().sum() == pytest.approx(0.0, abs=TOL)
    large = solve_day(inputs([50], pv=[1], id_price=[60], soc_start=1.0), cfg)
    # each position is bounded by the POI envelope, so the whole export goes to ID
    assert large.schedule["x_id_mw"].iloc[0] == pytest.approx(1.0)
    assert large.schedule["x_da_mw"].iloc[0] == pytest.approx(0.0, abs=TOL)


def test_market_positions_back_the_poi_flow(config_factory):
    cfg = config_factory(id_enabled=True, id_cost=0.1)
    rng = np.random.default_rng(3)
    da = rng.normal(50, 30, 8)
    res = solve_day(inputs(da, id_price=da + rng.normal(0, 10, 8), load=np.full(8, 0.5), pv=np.full(8, 1.0)), cfg)
    s = res.schedule
    assert np.allclose(s["x_da_mw"] + s["x_id_mw"], -s["p_poi_mw"], atol=TOL)
