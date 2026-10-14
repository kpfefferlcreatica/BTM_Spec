"""Daily deterministic model: battery, PV, load, DA, ID, peak, network charges,
aFRR capacity and aFRR energy.

Frames and signs follow 00-overview/02-conventions.md:
  p_poi > 0 import, p_poi < 0 export; market quantity > 0 sale.
  Bridge: p_poi = load - pv_out - (p_discharge - p_charge).
  Backing: x_da + x_id + e_up - e_dn = -p_poi (the whole POI flow is traded or
  activated; no retail contract).
  SOC: soc[t+1] = soc[t] + dt * (eta_c * p_charge - p_discharge / eta_d).

Each market position is bounded by the POI envelope. With foresight on DA and the
ID1 index this still permits DA/ID arbitrage up to that envelope; the ID
transaction cost is the only thing that dampens it.

aFRR capacity (ADR-018): the award is deterministic, `bid <= realised DE marginal
price`, and the whole offered volume of an awarded bid is accepted (price-taker:
partial acceptance of the marginal bid is ignored). Offered MW are integers
(1 MW minimum and step). The SOC corridor of `corridor_duration_slots` binds on
the awarded MW.

aFRR energy: per slot and direction one bid level is chosen from the levels of
`afrr_energy.activation_levels`, with an integer offered volume y. An award
obliges y >= r (ADR-018 §9); without free bids y == r. Activation
e = alpha * y is known (deterministic), enters the SOC balance and the POI flow,
and is paid via rho. Headroom at battery and POI binds on the offered volume y
on top of the schedule without activation, because activation can reach y
within the slot. The import peak is measured on the slot average, activation
included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from pf_bound.afrr_energy import SlotLevels, activation_levels
from pf_bound.config import SiteConfig

SLOT_HOURS = 0.25
UP, DN = 0, 1  # aFRR direction axis: POS, NEG


@dataclass(frozen=True)
class AfrrCapacityDay:
    block_of_slot: np.ndarray  # int [H], product block of each slot
    marginal_price: np.ndarray  # EUR/MW/h [B, 2], realised DE marginal capacity price
    history: np.ndarray  # EUR/MW/h [N, B, 2], marginal prices of the lookback days

    @property
    def blocks(self) -> int:
        return self.marginal_price.shape[0]


@dataclass(frozen=True)
class AfrrEnergyDay:
    cbmp_up: np.ndarray  # EUR/MWh [H, C], per optimisation cycle
    cbmp_dn: np.ndarray  # EUR/MWh [H, C]
    direction: np.ndarray  # int [H, C], sign of the netted DE setpoint: +1 up, -1 down, 0 none


@dataclass(frozen=True)
class DayInputs:
    da_price: np.ndarray  # EUR/MWh, [H]
    id_price: np.ndarray | None  # EUR/MWh, [H]; None when ID data is absent
    load_mw: np.ndarray  # [H], site consumption incl. auxiliary load
    pv_avail_mw: np.ndarray  # [H], available generation before curtailment
    soc_start_mwh: float
    peak_in_mw: float  # running import peak of the accounting period so far
    terminal_value_eur_per_mwh: float
    afrr_capacity: AfrrCapacityDay | None = None
    afrr_energy: AfrrEnergyDay | None = None

    @property
    def horizon(self) -> int:
        return len(self.da_price)


@dataclass(frozen=True)
class DayResult:
    objective_eur: float
    pnl: dict[str, float]
    schedule: pd.DataFrame
    soc_end_mwh: float
    peak_out_mw: float
    capacity_bids: np.ndarray | None = None  # EUR/MW/h [B, 2], nan = no bid
    awarded: np.ndarray | None = None  # bool [B, 2]
    reserve_mw: np.ndarray | None = None  # [B, 2]
    # integer decisions, kept so the shadow-value LP can fix them
    charging_mode: np.ndarray | None = field(default=None, repr=False)
    energy_level: dict | None = field(default=None, repr=False)  # direction -> chosen level index [H]


def solve_day(
    inp: DayInputs,
    cfg: SiteConfig,
    *,
    capacity_bids: np.ndarray | None = None,
    verbose: bool = False,
) -> DayResult:
    awarded = _awards(inp, capacity_bids)
    built = _build(inp, cfg, capacity_bids=capacity_bids, awarded=awarded)
    m, v, terms = built.model, built.v, built.terms
    m.Params.OutputFlag = int(verbose)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"daily model not optimal: status {m.Status}")

    h = inp.horizon
    poi_flow = v["imp"].X - v["exp"].X
    onehot = _onehot(inp)
    reserve = np.rint(v["r"].X) if v["r"] is not None else np.zeros((onehot.shape[1], 2))
    columns = {
        "p_charge_mw": v["pc"].X,
        "p_discharge_mw": v["pd"].X,
        "soc_start_mwh": v["soc"].X[:-1],
        "pv_out_mw": inp.pv_avail_mw - v["curt"].X,
        "pv_curtailed_mw": v["curt"].X,
        "p_poi_mw": poi_flow,
        "x_da_mw": v["x_da"].X,
        "x_id_mw": v["id_sell"].X - v["id_buy"].X,
        "r_up_mw": onehot @ reserve[:, UP],
        "r_dn_mw": onehot @ reserve[:, DN],
    }
    energy_level = None
    if v["energy"] is not None:
        energy_level = {}
        for d, tag in ((UP, "up"), (DN, "dn")):
            en = v["energy"][d]
            chosen = np.array([int(np.argmax(w.X)) for w in en["w"]])
            energy_level[d] = chosen
            columns[f"y_{tag}_mw"] = np.array([y.X for y in en["y"]])
            columns[f"e_{tag}_mw"] = np.array([e.getValue() for e in en["e"]])
            columns[f"beta_{tag}"] = np.array([en["levels"][t].beta[chosen[t]] for t in range(h)])
    return DayResult(
        objective_eur=m.ObjVal,
        pnl={k: _value(t) for k, t in terms.items()},
        schedule=pd.DataFrame(columns),
        soc_end_mwh=float(v["soc"].X[-1]),
        peak_out_mw=float(max(inp.peak_in_mw, poi_flow.max())),
        capacity_bids=capacity_bids,
        awarded=awarded,
        reserve_mw=reserve if v["r"] is not None else None,
        charging_mode=np.rint(v["mode"].X) if v["mode"] is not None else None,
        energy_level=energy_level,
    )


def capacity_shadow_values(inp: DayInputs, cfg: SiteConfig, res: DayResult) -> np.ndarray:
    """Reservation price mu [B, 2] in EUR/MW/h: spot-side value lost per MW of reserve.

    One-step difference on the LP with every discrete choice of `res` fixed
    (charging mode, energy bid level); reserve and offered energy volume become
    continuous. It is the cost of the last offered MW, or of the first MW where
    nothing was offered. LP duals are not used because they are not unique at
    degenerate points, and r = 0 with the battery at a limit is the typical case.
    +inf where one more MW is infeasible.
    """
    cap = inp.afrr_capacity
    base = np.zeros((cap.blocks, 2)) if res.reserve_mw is None else res.reserve_mw
    hours = _onehot(inp).sum(axis=0) * SLOT_HOURS
    built = _build(inp, cfg, fixed=res, fixed_reserve=base)
    m, fix = built.model, built.v["r_fix"]
    m.Params.OutputFlag = 0

    def solve(b: int, d: int, value: float) -> float:
        fix[b, d].RHS = value
        m.optimize()
        obj = m.ObjVal if m.Status == GRB.OPTIMAL else -np.inf
        fix[b, d].RHS = base[b, d]
        return obj

    mu = np.zeros((cap.blocks, 2))
    for b in range(cap.blocks):
        for d in (UP, DN):
            r = base[b, d]
            lo, hi = (r - 1, r) if r >= 1 else (r, r + 1)
            gap = solve(b, d, lo) - solve(b, d, hi)
            mu[b, d] = gap / hours[b] if np.isfinite(gap) else np.inf
    return mu


def _awards(inp: DayInputs, bids: np.ndarray | None) -> np.ndarray | None:
    if inp.afrr_capacity is None or bids is None:
        return None
    with np.errstate(invalid="ignore"):
        return ~np.isnan(bids) & (bids <= inp.afrr_capacity.marginal_price)


def _onehot(inp: DayInputs) -> np.ndarray:
    if inp.afrr_capacity is None:
        return np.zeros((inp.horizon, 0))
    cap = inp.afrr_capacity
    return (cap.block_of_slot[:, None] == np.arange(cap.blocks)[None, :]).astype(float)


@dataclass
class _Built:
    model: gp.Model
    terms: dict
    v: dict


def _build(
    inp: DayInputs,
    cfg: SiteConfig,
    *,
    capacity_bids: np.ndarray | None = None,
    awarded: np.ndarray | None = None,
    fixed: DayResult | None = None,
    fixed_reserve: np.ndarray | None = None,
) -> _Built:
    """`fixed` turns the MILP into the LP used for shadow values: every discrete
    choice of that result is fixed and reserve is pinned by `fixed_reserve`."""
    h, dt = inp.horizon, SLOT_HOURS
    b, poi, tar = cfg.battery, cfg.poi, cfg.tariff
    id_on = cfg.market.id_enabled and inp.id_price is not None
    int_type = GRB.CONTINUOUS if fixed is not None else GRB.INTEGER

    m = gp.Model("pf_day")
    m.Params.OutputFlag = 0

    pc = m.addMVar(h, lb=0, ub=b.p_charge_max_mw, name="p_charge")
    pd_ = m.addMVar(h, lb=0, ub=b.p_discharge_max_mw, name="p_discharge")
    soc = m.addMVar(h + 1, lb=b.soc_min_mwh, ub=b.soc_max_mwh, name="soc")
    curt = m.addMVar(h, lb=0, ub=inp.pv_avail_mw, name="pv_curtailed")
    imp = m.addMVar(h, lb=0, ub=poi.import_limit_mw, name="import")
    exp = m.addMVar(h, lb=0, ub=poi.export_limit_mw, name="export")
    x_da = m.addMVar(h, lb=-poi.import_limit_mw, ub=poi.export_limit_mw, name="x_da")
    id_ub = (poi.export_limit_mw, poi.import_limit_mw) if id_on else (0.0, 0.0)
    id_sell = m.addMVar(h, lb=0, ub=id_ub[0], name="x_id_sell")
    id_buy = m.addMVar(h, lb=0, ub=id_ub[1], name="x_id_buy")
    z = m.addVar(lb=inp.peak_in_mw, name="peak")

    p_poi = imp - exp
    p_batt = pd_ - pc
    m.addConstr(soc[0] == inp.soc_start_mwh)
    m.addConstr(soc[1:] == soc[:-1] + dt * (b.eta_charge * pc - pd_ / b.eta_discharge), name="soc_balance")
    m.addConstr(p_poi == inp.load_mw - (inp.pv_avail_mw - curt) - p_batt, name="bridge")
    m.addConstr(z >= p_poi, name="peak_epigraph")

    if cfg.degradation.cycle_limit_per_day is not None:
        m.addConstr(dt * pd_.sum() <= cfg.degradation.cycle_limit_per_day * b.energy_capacity_mwh, name="cycle_limit")

    mode = None
    if cfg.solver.forbid_simultaneous_charge_discharge:
        if fixed is None:
            mode = m.addMVar(h, vtype=GRB.BINARY, name="charging")
            charging = mode
        else:
            charging = fixed.charging_mode
        m.addConstr(pc <= b.p_charge_max_mw * charging)
        m.addConstr(pd_ <= b.p_discharge_max_mw * (1 - charging))

    id_price = inp.id_price if id_on else np.zeros(h)
    terms = {
        "da": dt * (inp.da_price @ x_da),
        "id": dt * (id_price @ (id_sell - id_buy)),
        "id_transaction_cost": -dt * cfg.market.id_transaction_cost_eur_per_mwh * (id_sell.sum() + id_buy.sum()),
        "network_volumetric": -dt * tar.volumetric_charge_eur_per_mwh * imp.sum(),
        "levies": -dt * tar.levies_eur_per_mwh * imp.sum(),
        "peak": -tar.peak_price_eur_per_mw * (z - inp.peak_in_mw),
        "terminal": inp.terminal_value_eur_per_mwh * soc[h],
        "afrr_capacity": 0.0,
        "afrr_energy": 0.0,
    }
    r_max = b.p_charge_max_mw + b.p_discharge_max_mw

    # reserve per slot and direction; zero without capacity data
    r = r_fix = None
    r_slot = {UP: np.zeros(h), DN: np.zeros(h)}
    if inp.afrr_capacity is not None:
        onehot = _onehot(inp)
        nb = onehot.shape[1]
        if fixed is None:
            ub = np.where(awarded, r_max, 0.0) if awarded is not None else np.zeros((nb, 2))
            r = m.addMVar((nb, 2), lb=0, ub=ub, vtype=GRB.INTEGER, name="r")
            hours = onehot.sum(axis=0) * dt
            price = np.where(awarded, capacity_bids, 0.0) if awarded is not None else np.zeros((nb, 2))
            terms["afrr_capacity"] = gp.quicksum(
                price[i, d] * hours[i] * r[i, d] for i in range(nb) for d in (UP, DN) if price[i, d] != 0
            )
        else:
            r = m.addMVar((nb, 2), lb=-GRB.INFINITY, name="r")
            r_fix = np.array([[m.addConstr(r[i, d] == fixed_reserve[i, d]) for d in (UP, DN)] for i in range(nb)])
        r_slot = {UP: onehot @ r[:, UP], DN: onehot @ r[:, DN]}
        dc = cfg.afrr.corridor_duration_slots * dt
        for s in (soc[:-1], soc[1:]):
            m.addConstr(s >= b.soc_min_mwh + r_slot[UP] * dc / b.eta_discharge, name="corridor_up")
            m.addConstr(s <= b.soc_max_mwh - r_slot[DN] * dc * b.eta_charge, name="corridor_dn")

    # offered energy volume per slot and direction; equals reserve without energy data
    energy = None
    e = {UP: 0.0, DN: 0.0}
    offer = r_slot
    if inp.afrr_energy is not None:
        en_in = inp.afrr_energy
        energy, offer, e = {}, {}, {}
        for d, cbmp in ((UP, en_in.cbmp_up), (DN, en_in.cbmp_dn)):
            levels = activation_levels(cbmp, en_in.direction, up=(d == UP))
            ys, ws, es, revenue = [], [], [], []
            for t, lv in enumerate(levels):
                k = len(lv.beta)
                if fixed is None:
                    w = m.addMVar(k, vtype=GRB.BINARY, name=f"w_{d}_{t}")
                else:
                    chosen = np.zeros(k)
                    chosen[fixed.energy_level[d][t]] = 1.0
                    w = m.addMVar(k, lb=chosen, ub=chosen, name=f"w_{d}_{t}")
                yk = m.addMVar(k, lb=0, ub=r_max, name=f"yk_{d}_{t}")
                y = m.addVar(lb=0, ub=r_max, vtype=int_type, name=f"y_{d}_{t}")
                m.addConstr(w.sum() == 1)
                m.addConstr(yk <= r_max * w)
                m.addConstr(y == yk.sum())
                ys.append(y)
                ws.append(w)
                es.append(lv.alpha @ yk)
                revenue.append(lv.rho @ yk)
            y_vec = gp.MVar.fromlist(ys)
            e_vec = gp.quicksum(alpha_term * unit for alpha_term, unit in zip(es, _unit_vectors(h)))
            m.addConstr(y_vec >= r_slot[d], name=f"energy_offer_obligation_{d}")
            if not cfg.afrr.free_energy_bids:
                m.addConstr(y_vec <= r_slot[d], name=f"no_free_energy_bids_{d}")
            sign = 1.0 if d == UP else -1.0  # down: rho is what the BSP pays
            terms["afrr_energy"] = terms["afrr_energy"] + sign * dt * gp.quicksum(revenue)
            energy[d] = {"y": ys, "w": ws, "e": es, "levels": levels}
            offer[d], e[d] = y_vec, e_vec

    m.addConstr(x_da + id_sell - id_buy + e[UP] - e[DN] == -p_poi, name="backing")
    p_batt_base = p_batt - e[UP] + e[DN]
    p_poi_base = p_poi + e[UP] - e[DN]
    if inp.afrr_capacity is not None or inp.afrr_energy is not None:
        m.addConstr(p_batt_base + offer[UP] <= b.p_discharge_max_mw, name="headroom_batt_up")
        m.addConstr(p_batt_base - offer[DN] >= -b.p_charge_max_mw, name="headroom_batt_dn")
        m.addConstr(p_poi_base - offer[UP] >= -poi.export_limit_mw, name="headroom_poi_up")
        m.addConstr(p_poi_base + offer[DN] <= poi.import_limit_mw, name="headroom_poi_dn")

    m.setObjective(gp.quicksum(terms.values()), GRB.MAXIMIZE)
    v = dict(pc=pc, pd=pd_, soc=soc, curt=curt, imp=imp, exp=exp, x_da=x_da, id_sell=id_sell, id_buy=id_buy,
             r=r if fixed is None else None, r_fix=r_fix, mode=mode, energy=energy)
    return _Built(m, terms, v)


def _unit_vectors(h: int):
    for t in range(h):
        unit = np.zeros(h)
        unit[t] = 1.0
        yield unit


def _value(term) -> float:
    if isinstance(term, (int, float)):
        return float(term)
    value = term.getValue()
    return float(np.asarray(value).item())
