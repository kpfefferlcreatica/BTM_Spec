# pf_bound — policy-restricted foresight bound

Throwaway prototype, **not normative**. It answers design questions for the
Planner's foresight benchmark (T5 §4); the specification remains the authority.

## What it models

One deterministic MILP per day with realised spot, load and PV. The plan may use
the known aFRR award and activation; that optimism is accepted and labelled. Bids
are not optimised with foresight: the capacity price follows the ADR-018 policy
against 30 days of DE marginal capacity prices, and the energy bid is a single
`β` per 15-min product and direction.

| Piece | Status |
|---|---|
| Battery, SOC, PV curtailment, bridge, backing | done |
| DA (15 min), ID1 index with transaction cost | done |
| Running peak per accounting period, volumetric charges, levies | done |
| Daily cycle limit, linear terminal value `V` from D+1 DA | done |
| aFRR capacity: award, ADR-018 bid policy, fixed point on `μ` | done |
| aFRR energy: 4-s cycles → in-merit setpoint → `α`, `ρ`; one `β` per slot and direction; offer obligation | done |
| Day-to-day driver (SOC, running peak, `V` from D+1, lookback to D-2), P&L and Parquet export | done |
| Data fetchers and loaders (regelleistung.net, ENTSO-E, netztransparenz, EPEX, load/PV) → `DayData` | next; no data downloaded yet |
| Delineation (ADR-017) | out of first build |

Deliberate departures from the real Planner: hard SOC bounds, no imbalance
variable, risk-neutral. The whole POI flow is assumed to be traded (no retail
contract for site load).

## Run

```
pip install -e '.[test]'
pytest
```

The pip-installed Gurobi licence is size-limited (~2000 variables), enough for the
tests. Full days need the full licence (`grbgetkey`).

Raw and processed data belong in `data/`, which is git-ignored.
