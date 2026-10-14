"""aFRR energy: per-cycle activation from realised CBMP and setpoint direction,
the single energy bid level per slot and direction, the offer obligation, and
the physical and settlement effects of activation."""

from dataclasses import replace

import numpy as np
import pytest

from pf_bound.afrr_energy import activation_levels
from pf_bound.day_model import AfrrCapacityDay, AfrrEnergyDay, DayInputs, solve_day

TOL = 1e-6
NONE = np.nan


def test_up_levels_from_cycles_with_up_demand_only():
    direction = np.array([[+1, +1, 0, -1]])
    cbmp_up = np.array([[10.0, 30.0, 99.0, 99.0]])
    (lv,) = activation_levels(cbmp_up, direction, up=True)
    # none, beta=30 (cycle 2), beta=10 (cycles 1-2)
    assert np.isnan(lv.beta[0])
    assert lv.beta[1:].tolist() == [30.0, 10.0]
    assert lv.alpha.tolist() == pytest.approx([0.0, 0.25, 0.5])
    assert lv.rho.tolist() == pytest.approx([0.0, 7.5, 10.0])


def test_down_levels_activate_at_or_below_bid():
    direction = np.array([[-1, -1, +1, 0]])
    cbmp_dn = np.array([[5.0, -20.0, -99.0, -99.0]])
    (lv,) = activation_levels(cbmp_dn, direction, up=False)
    # none, beta=-20 (cycle 2), beta=5 (cycles 1-2); rho is the price the BSP pays per MW
    assert lv.beta[1:].tolist() == [-20.0, 5.0]
    assert lv.alpha.tolist() == pytest.approx([0.0, 0.25, 0.5])
    assert lv.rho.tolist() == pytest.approx([0.0, -5.0, -3.75])


def test_slot_without_demand_has_only_the_none_level():
    (lv,) = activation_levels(np.array([[50.0, 60.0]]), np.array([[0, -1]]), up=True)
    assert lv.alpha.tolist() == [0.0]


def energy_day(*, h=1, cycles=4, cbmp_up=0.0, cbmp_dn=0.0, direction=0, soc_start=0.5, da=0.0,
               capacity=None):
    full = lambda x: np.broadcast_to(np.asarray(x, float), (h, cycles)).copy()
    return DayInputs(
        da_price=np.full(h, da, float),
        id_price=None,
        load_mw=np.zeros(h),
        pv_avail_mw=np.zeros(h),
        soc_start_mwh=soc_start,
        peak_in_mw=0.0,
        terminal_value_eur_per_mwh=0.0,
        afrr_capacity=capacity,
        afrr_energy=AfrrEnergyDay(
            cbmp_up=full(cbmp_up), cbmp_dn=full(cbmp_dn), direction=full(direction).astype(int)
        ),
    )


def test_free_up_bid_uses_the_full_swing(config_factory):
    # always in merit at 100. Base schedule: charge 4 MW bought on DA at 0; activation 8 MW
    # swings the battery to +4 MW discharge, which empties the full battery in the slot
    cfg = config_factory(eta=1.0, free_energy_bids=True)
    res = solve_day(energy_day(cbmp_up=100.0, direction=+1, soc_start=1.0), cfg)
    s = res.schedule
    assert s["y_up_mw"].iloc[0] == pytest.approx(8.0)
    assert s["e_up_mw"].iloc[0] == pytest.approx(8.0)
    assert s["beta_up"].iloc[0] == pytest.approx(100.0)
    assert s["x_da_mw"].iloc[0] == pytest.approx(-4.0)
    assert s["p_discharge_mw"].iloc[0] == pytest.approx(4.0)
    assert res.pnl["afrr_energy"] == pytest.approx(100.0 * 8 * 0.25)
    assert res.soc_end_mwh == pytest.approx(0.0, abs=TOL)


def test_no_energy_offer_without_award_or_free_bids(config_factory):
    cfg = config_factory(eta=1.0, free_energy_bids=False)
    res = solve_day(energy_day(cbmp_up=100.0, direction=+1, soc_start=1.0), cfg)
    assert res.schedule["y_up_mw"].iloc[0] == pytest.approx(0.0, abs=TOL)
    assert res.pnl["afrr_energy"] == pytest.approx(0.0, abs=TOL)


def test_award_obliges_an_offer_that_may_stay_out_of_merit(config_factory):
    # energy is worth 100 (DA and terminal), so activation at CBMP 10 is a loss and the obliged
    # offer is placed out of merit. A full battery has no charging swing: r = 4
    cfg = config_factory(eta=1.0, corridor_slots=0, free_energy_bids=False)
    capacity = AfrrCapacityDay(
        block_of_slot=np.zeros(2, int), marginal_price=np.array([[50.0, 0.0]]), history=np.zeros((0, 1, 2))
    )
    inp = replace(energy_day(h=2, cbmp_up=10.0, direction=+1, soc_start=1.0, da=100.0, capacity=capacity),
                  terminal_value_eur_per_mwh=100.0)
    res = solve_day(inp, cfg, capacity_bids=np.array([[1.0, np.nan]]))
    s = res.schedule
    assert s["r_up_mw"].tolist() == pytest.approx([4.0, 4.0])
    assert (s["y_up_mw"] >= 4.0 - TOL).all()
    assert s["e_up_mw"].sum() == pytest.approx(0.0, abs=TOL)
    assert s["beta_up"].isna().all()


def test_down_activation_at_negative_cbmp_earns_and_charges(config_factory):
    # BSP is paid 50 EUR/MWh to absorb; base schedule discharges 4 MW sold on DA at 0,
    # activation 8 MW swings the empty battery to 4 MW charging
    cfg = config_factory(eta=1.0, free_energy_bids=True)
    res = solve_day(energy_day(cbmp_dn=-50.0, direction=-1, soc_start=0.0), cfg)
    assert res.schedule["e_dn_mw"].iloc[0] == pytest.approx(8.0)
    assert res.pnl["afrr_energy"] == pytest.approx(50.0 * 8 * 0.25)
    assert res.soc_end_mwh == pytest.approx(1.0)


def test_activation_backs_the_poi_flow_with_spot(config_factory):
    cfg = config_factory(eta=0.9, free_energy_bids=True)
    rng = np.random.default_rng(5)
    h, c = 4, 8
    inp = DayInputs(
        da_price=rng.normal(50, 30, h), id_price=None, load_mw=np.full(h, 0.3), pv_avail_mw=np.full(h, 0.5),
        soc_start_mwh=0.5, peak_in_mw=0.0, terminal_value_eur_per_mwh=0.0,
        afrr_energy=AfrrEnergyDay(
            cbmp_up=rng.normal(80, 40, (h, c)), cbmp_dn=rng.normal(20, 40, (h, c)),
            direction=rng.choice([-1, 0, 1], (h, c)),
        ),
    )
    s = solve_day(inp, cfg).schedule
    assert np.allclose(s["x_da_mw"] + s["x_id_mw"] + s["e_up_mw"] - s["e_dn_mw"], -s["p_poi_mw"], atol=TOL)


def test_headroom_binds_on_offered_volume_over_the_base_schedule(config_factory):
    cfg = config_factory(eta=0.9, free_energy_bids=True)
    rng = np.random.default_rng(11)
    h, c = 4, 8
    inp = DayInputs(
        da_price=rng.normal(50, 30, h), id_price=None, load_mw=np.zeros(h), pv_avail_mw=np.zeros(h),
        soc_start_mwh=0.5, peak_in_mw=0.0, terminal_value_eur_per_mwh=0.0,
        afrr_energy=AfrrEnergyDay(
            cbmp_up=rng.normal(120, 40, (h, c)), cbmp_dn=rng.normal(-20, 40, (h, c)),
            direction=rng.choice([-1, 0, 1], (h, c)),
        ),
    )
    s = solve_day(inp, cfg).schedule
    base = s["p_discharge_mw"] - s["p_charge_mw"] - s["e_up_mw"] + s["e_dn_mw"]
    assert (base + s["y_up_mw"] <= 4.0 + TOL).all()
    assert (base - s["y_dn_mw"] >= -4.0 - TOL).all()
    assert s["y_up_mw"].sum() + s["y_dn_mw"].sum() > 0


def test_bid_policy_fixed_point_runs_with_energy(config_factory):
    from pf_bound.bid_policy import solve_day_with_bid_policy

    cfg = config_factory(eta=0.9, corridor_slots=1, free_energy_bids=False)
    rng = np.random.default_rng(2)
    h, c = 4, 8
    inp = DayInputs(
        da_price=rng.normal(60, 30, h), id_price=None, load_mw=np.zeros(h), pv_avail_mw=np.zeros(h),
        soc_start_mwh=0.5, peak_in_mw=0.0, terminal_value_eur_per_mwh=50.0,
        afrr_capacity=AfrrCapacityDay(
            block_of_slot=np.array([0, 0, 1, 1]),
            marginal_price=np.array([[30.0, 25.0], [5.0, 40.0]]),
            history=rng.uniform(0, 50, (10, 2, 2)),
        ),
        afrr_energy=AfrrEnergyDay(
            cbmp_up=rng.normal(120, 40, (h, c)), cbmp_dn=rng.normal(-20, 40, (h, c)),
            direction=rng.choice([-1, 0, 1], (h, c)),
        ),
    )
    out = solve_day_with_bid_policy(inp, cfg)
    s = out.result.schedule
    assert (s["y_up_mw"] == s["r_up_mw"]).all() and (s["y_dn_mw"] == s["r_dn_mw"]).all()
    assert out.result.objective_eur == pytest.approx(sum(out.result.pnl.values()))
    assert 1 <= len(out.log) <= cfg.afrr.bid_policy_max_iter
