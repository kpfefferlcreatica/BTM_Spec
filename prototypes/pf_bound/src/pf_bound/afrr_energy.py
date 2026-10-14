"""Zone -> asset aFRR activation mapping from realised CBMP and setpoint direction.

Per optimisation cycle (4 s, equal weight) a bid in direction `up` is in merit iff
the netted German setpoint asks for that direction and
  up:   CBMP_up >= beta
  down: CBMP_dn <= beta
Under the price-taker assumption an in-merit bid is activated in full, and with
ideal delivery the asset follows the stepped setpoint exactly; the 30 s + 270 s
acceptance channel is a tolerance for slow units, not the setpoint shape, so it
is not applied. Settlement is per second at max(beta, CBMP) up and min(beta, CBMP)
down, which equals the CBMP whenever the bid is in merit.

The same mapping serves the bound and backtest settlement. Because activation
only changes at realised CBMP values, the candidate bid levels of a slot are
exactly those values (plus `none`, never activated), which is lossless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlotLevels:
    beta: np.ndarray  # EUR/MWh [K]; beta[0] = nan, the never-activated level
    alpha: np.ndarray  # fraction of the slot activated [K], per MW offered
    rho: np.ndarray  # mean over cycles of activation * CBMP [K], EUR/MWh per MW offered


def activation_levels(cbmp: np.ndarray, direction: np.ndarray, *, up: bool) -> list[SlotLevels]:
    """cbmp, direction: [H, C]. direction +1 up demand, -1 down demand, 0 none."""
    cbmp = np.asarray(cbmp, float)
    want = +1 if up else -1
    levels = []
    for prices, dirs in zip(cbmp, np.asarray(direction)):
        n = len(prices)
        active = prices[dirs == want]
        # order from the least to the most often activated bid
        beta = np.unique(active)[::-1] if up else np.unique(active)
        in_merit = active[None, :] >= beta[:, None] if up else active[None, :] <= beta[:, None]
        alpha = in_merit.sum(axis=1) / n
        rho = (in_merit * active[None, :]).sum(axis=1) / n
        levels.append(
            SlotLevels(
                beta=np.concatenate([[np.nan], beta]),
                alpha=np.concatenate([[0.0], alpha]),
                rho=np.concatenate([[0.0], rho]),
            )
        )
    return levels
