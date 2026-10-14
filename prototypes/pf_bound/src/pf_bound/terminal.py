"""Terminal value of stored energy at the end of a daily solve.

V = eta_d * mean(price of the n most expensive slots of D+1), where n is the
number of slots needed to discharge the usable SOC band at full power. Linear in
soc_end by construction; the foresight bound reads D+1 realised DA prices.
"""

from __future__ import annotations

import math

import numpy as np

from pf_bound.config import SiteConfig


def terminal_value(next_day_prices, cfg: SiteConfig, slot_hours: float = 0.25) -> float:
    b = cfg.battery
    prices = np.sort(np.asarray(next_day_prices, float))[::-1]
    ac_energy = (b.soc_max_mwh - b.soc_min_mwh) * b.eta_discharge
    n = math.ceil(ac_energy / (b.p_discharge_max_mw * slot_hours) - 1e-9)
    n = max(1, min(n, len(prices)))
    return float(b.eta_discharge * prices[:n].mean())
