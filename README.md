# BTM Engine — Architecture Specification

A cross-market optimisation engine for a behind-the-meter battery, co-optimising
day-ahead spot, continuous intraday, and aFRR (capacity + energy) against peak
shaving, load shifting and arbitrage, under asset, POI and German grid-fee
constraints.

**This repository contains no implementation.** It contains the normative
contracts, the layer designs, the compliance architecture and the sequencing
plan. Implementation happens in the engine repository against these documents.

---

## Status of this specification

| Part | Status |
|---|---|
| Layer model and clocks | Normative |
| Units, signs, time grid | Normative |
| Glossary | Normative for what a term means; never for what the thing does. §14 carries the vocabulary rulings |
| Delineation regime (MiSpel, A1/A5) | Normative, transcribed from Anlage 1; route split and identities derived |
| Delineation in the objective (ADR-017) | Normative — accumulators on C5/C6, state equations in L3, exact recomputation in L5 |
| ADR-001 … ADR-014, ADR-016 … ADR-020 | Accepted |
| ADR-015 (open register) | Placeholder — its three decisions are named below |
| Seam contracts C1–C6 | Normative, versioned |
| Layer designs L0–L5 | Normative for structure, indicative for algorithm choice |
| Compliance architecture | Normative |
| Implementation plan | Advisory |
| Technical notes (`06-theory/`) | Advisory — builder output. A note's verdicts bind nothing until an ADR acts on them |

**Status vocabulary.** Two values:

| Status | Meaning |
|---|---|
| **Normative** | Binding as written. |
| **Advisory** | Guidance; may be departed from with reason. |

Consistency between documents is held by the ADRs and by review.

Three decisions are deliberately **deferred**: the MILP solver (Gurobi
intended), the per-tick latency budget, and intraday fill-model fidelity.
Nothing in this specification depends on them. Where a document must reference
one, it references the *abstraction*, never the choice. `01-adr/ADR-015` is a
title-only placeholder: the three return only if a technical note acquires a
dependency on one. `05-implementation/P0-workstreams.md` carries the per-workstream
review that checks no such dependency has appeared.

---

## How to read this

Read in this order. Each part assumes the previous one.

1. `00-overview/01-system-model.md` — the layer model, the two feedback edges
   and how they are resolved, the four clocks. **Start here.**
2. `00-overview/02-conventions.md` — units, sign conventions, the time grid,
   naming. Every other document depends on these being unambiguous.
3. `00-overview/03-mispel-reference.md` — the MiSpel delineation machinery
   (Abgrenzungsoption, cases A1 and A5) in the regulator's own notation.
4. `00-overview/04-glossary.md` — what each term means and which document owns
   it. Keep it open while reading the rest; §14 records the words that carried
   two meanings and which meaning kept each one.
5. `01-adr/` — the decisions that are expensive to reverse, each with context,
   decision, consequences and the rejected alternatives.
6. `03-contracts/` — the seven seams. These are the frozen surface. If you read
   only one section, read `C0-contract-conventions` and `C2-valuation-to-planner`.
7. `02-layers/` — the internals of each layer.
8. `04-compliance/` — how we know it works: the seven test levels.
9. `05-implementation/` — sequencing and the agent playbook.
10. `06-theory/` — the technical notes. `TN-NN` numbering is sequential and
   never recycled, but it is per-repository: a note carrying the same number on
   a research branch is a different note. Read a note as the builder's argument,
   not as a verdict the specification has accepted.

---

## The one-paragraph summary

The engine is a strictly layered pipeline. **Belief** is a bitemporal,
content-addressed view over market and site data that can physically not return
information the engine did not yet know. **Valuation** turns beliefs into
*linearizable economic primitives* — piecewise-linear curves, epigraph
coefficients, bounds — never into scalar prices, because peak and reserve terms
are not linear in the decision. **Planner** composes those primitives into a
single MILP family solved at gates, carrying a commitment ledger
forward, and emits order *intent*. **Execution** is the pre-existing market
simulation and is treated as an external system behind a thin adapter.
**Settlement** recomputes the truth ex post and decomposes the gap between
planned and realised value into four attributable buckets. A sixth component,
the **State/Value store**, holds everything that must survive a tick — realised
peak, tariff qualification state, and the end-of-horizon value function V(SOC) —
and is the only path by which information flows backwards, always with a
one-tick lag.

---

## Repository map

```
00-overview/     system model, conventions, MiSpel reference, glossary
01-adr/          architecture decision records (stable IDs, referenced everywhere)
02-layers/       per-layer internal design (L0-L5)
03-contracts/    the seven seams — the frozen surface (C0 conventions, C1-C6)
04-compliance/   invariants, property tests, replay, gap instrumentation (T0-T6)
05-implementation/ workstreams, sequencing, agent playbook
06-theory/        technical notes: the analytic work behind a decision
stubs/           C# interface stubs — signatures only, no bodies
prototypes/      throwaway Python prototypes — not normative (pf_bound: foresight bound)
```

## The five architectural corrections

The original five-layer framing was sound. Five things in it did not survive
contact with the detail, and each is now an ADR:

1. **The graph had a cycle.** λ_SOC is a Planner dual and `pPoiRealisedPeakMw` is a
   Settlement output, yet both sat in Valuation. Resolved by L0 and a strict
   forward-within-a-tick, lagged-across-ticks rule (ADR-006).
2. **λ_SOC as a scalar misprices the curve.** Replaced by a concave piecewise
   linear V(SOC); λ becomes an *output*, the subgradient at the optimum (ADR-007).
3. **"Publish functions, not scalars" needed a form.** Five linearizable
   primitives, so Valuation can express non-linear economics and the Planner
   stays a MILP of statically known class (ADR-008).
4. **Hard pre-allocation of aFRR capacity is a primal restriction** — it deletes
   options and the loss is invisible. Replaced by a *dual price* for headroom,
   discovered by Lagrangian decomposition, with a certified gap (ADR-010).
5. **The annual maximum is not the only tariff regime.** §19(2) S.1 prices the
   peak within published HLZF windows rather than the annual maximum, and §19(2)
   S.2 intensive-use qualification is a cliff that battery operation moves
   directly, through both the numerator and the denominator of full-load hours.
   `PeakView` is a plug-in and qualification state is a dimension of V (ADR-011).

## Change discipline

A change to any file in `03-contracts/` requires, in the same commit:
its version bump, an update to the affected layer designs, and an update to the
corresponding conformance tests. This rule is the single most important process
constraint in the project and is restated in the engine repository's
`CLAUDE.md`.
