"""aFRR capacity bid policy (ADR-018) and its fixed point with the daily model.

p = argmax_p (p - mu) * P(award | p), p >= mu (INV-P-11), where P(award | p) is
the share of lookback days whose DE marginal price for the same block and
direction was >= p. P is a step function that only changes at historical
prices, so the maximiser is always one of them.

The realised marginal price never enters the bid; it only decides the award.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pf_bound.config import SiteConfig
from pf_bound.day_model import DN, UP, DayInputs, DayResult, capacity_shadow_values, solve_day


def award_probability(price: float, history: np.ndarray) -> float:
    history = np.asarray(history, float)
    return float(np.mean(history >= price)) if history.size else 0.0


def choose_bid_price(mu: float, history: np.ndarray) -> float | None:
    history = np.asarray(history, float)
    candidates = np.unique(history[history >= max(mu, 0.0)])
    if not np.isfinite(mu) or candidates.size == 0:
        return None
    value = [(p - mu) * award_probability(p, history) for p in candidates]
    return float(candidates[int(np.argmax(value))])


def policy_bids(mu: np.ndarray, history: np.ndarray) -> np.ndarray:
    bids = np.full(mu.shape, np.nan)
    for b in range(mu.shape[0]):
        for d in (UP, DN):
            p = choose_bid_price(mu[b, d], history[:, b, d])
            if p is not None:
                bids[b, d] = p
    return bids


@dataclass(frozen=True)
class BidPolicyOutcome:
    result: DayResult
    bids: np.ndarray  # EUR/MW/h [B, 2], nan = no bid
    mu: np.ndarray  # EUR/MW/h [B, 2] at the returned iterate
    converged: bool
    log: pd.DataFrame  # one row per iteration


def solve_day_with_bid_policy(inp: DayInputs, cfg: SiteConfig) -> BidPolicyOutcome:
    """Iterate solve -> mu -> bids until the bids repeat, at most bid_policy_max_iter
    solves. Without convergence the iterate with the highest objective is kept."""
    history = inp.afrr_capacity.history
    bids = policy_bids(np.zeros((inp.afrr_capacity.blocks, 2)), history)
    iterates, rows = [], []
    for k in range(cfg.afrr.bid_policy_max_iter):
        res = solve_day(inp, cfg, capacity_bids=bids)
        mu = capacity_shadow_values(inp, cfg, res)
        iterates.append((res, bids, mu))
        rows.append({"iteration": k, "objective_eur": res.objective_eur, "bids": bids.copy(),
                     "awarded": res.awarded.copy(), "mu": mu.copy()})
        new = policy_bids(mu, history)
        if np.array_equal(new, bids, equal_nan=True):
            return BidPolicyOutcome(res, bids, mu, True, pd.DataFrame(rows))
        bids = new
    res, bids, mu = max(iterates, key=lambda it: it[0].objective_eur)
    return BidPolicyOutcome(res, bids, mu, False, pd.DataFrame(rows))
