"""Site configuration loaded from YAML.

Values not yet known for a concrete site are tagged `!placeholder <value>`. The
loader records every tagged path and refuses the file unless the caller passes
`allow_placeholders=True`, so a placeholder can never reach a production run
unnoticed. Units follow 00-overview/02-conventions.md: MW, MWh, EUR/MWh, EUR/MW;
durations are slot counts.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class PlaceholderError(ValueError):
    pass


@dataclass(frozen=True)
class Battery:
    energy_capacity_mwh: float
    soc_min_mwh: float
    soc_max_mwh: float
    p_charge_max_mw: float
    p_discharge_max_mw: float
    eta_charge: float
    eta_discharge: float


@dataclass(frozen=True)
class Poi:
    import_limit_mw: float
    export_limit_mw: float


@dataclass(frozen=True)
class Tariff:
    peak_price_eur_per_mw: float
    peak_accounting_period: str  # "month" or "year"
    volumetric_charge_eur_per_mwh: float
    levies_eur_per_mwh: float


@dataclass(frozen=True)
class Market:
    id_enabled: bool
    id_transaction_cost_eur_per_mwh: float


@dataclass(frozen=True)
class Degradation:
    cycle_limit_per_day: float | None  # equivalent full cycles; None = no limit


@dataclass(frozen=True)
class Afrr:
    corridor_duration_slots: int
    free_energy_bids: bool
    capacity_lookback_days: int
    bid_policy_max_iter: int


@dataclass(frozen=True)
class Solver:
    forbid_simultaneous_charge_discharge: bool


@dataclass(frozen=True)
class SiteConfig:
    battery: Battery
    poi: Poi
    tariff: Tariff
    market: Market
    degradation: Degradation
    afrr: Afrr
    solver: Solver
    placeholders: tuple[str, ...] = ()


_SECTION_TYPES = {
    "battery": Battery,
    "poi": Poi,
    "tariff": Tariff,
    "market": Market,
    "degradation": Degradation,
    "afrr": Afrr,
    "solver": Solver,
}


class _Placeholder:
    def __init__(self, value: Any):
        self.value = value


class _Loader(yaml.SafeLoader):
    pass


def _placeholder(loader: yaml.SafeLoader, node: yaml.Node) -> _Placeholder:
    return _Placeholder(yaml.safe_load(loader.construct_scalar(node)))


_Loader.add_constructor("!placeholder", _placeholder)


def load_config(path: str | Path, allow_placeholders: bool = False) -> SiteConfig:
    raw = yaml.load(Path(path).read_text(), Loader=_Loader) or {}

    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")

    placeholders: list[str] = []
    sections = {}
    for name, cls in _SECTION_TYPES.items():
        if name not in raw:
            raise ValueError(f"missing config section: {name}")
        body = raw[name] or {}
        expected = {f.name for f in fields(cls)}
        if set(body) != expected:
            raise ValueError(
                f"section {name}: missing {sorted(expected - set(body))}, "
                f"unknown {sorted(set(body) - expected)}"
            )
        values = {}
        for key, value in body.items():
            if isinstance(value, _Placeholder):
                placeholders.append(f"{name}.{key}")
                value = value.value
            values[key] = value
        sections[name] = cls(**values)

    if placeholders and not allow_placeholders:
        raise PlaceholderError(f"placeholder values present: {', '.join(placeholders)}")

    cfg = SiteConfig(**sections, placeholders=tuple(placeholders))
    _validate(cfg)
    return cfg


def _validate(cfg: SiteConfig) -> None:
    b = cfg.battery
    if not 0 <= b.soc_min_mwh < b.soc_max_mwh <= b.energy_capacity_mwh:
        raise ValueError("require 0 <= soc_min_mwh < soc_max_mwh <= energy_capacity_mwh")
    if not (0 < b.eta_charge <= 1 and 0 < b.eta_discharge <= 1):
        raise ValueError("efficiencies must lie in (0, 1]")
    if min(b.p_charge_max_mw, b.p_discharge_max_mw, cfg.poi.import_limit_mw, cfg.poi.export_limit_mw) < 0:
        raise ValueError("power limits must be non-negative")
    if cfg.tariff.peak_accounting_period not in ("month", "year"):
        raise ValueError("peak_accounting_period must be 'month' or 'year'")
    if cfg.afrr.corridor_duration_slots < 0:
        raise ValueError("corridor_duration_slots must be >= 0")
