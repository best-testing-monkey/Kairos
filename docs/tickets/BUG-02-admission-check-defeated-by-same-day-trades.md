# BUG-02 — Margin admission gate silently disabled for accounts dominated by same-day round-trip trades

**Severity:** High (risk-control failure, not just a reporting bug). Consequence of BUG-01
(docs/tickets/BUG-01-same-day-fill-close-blind-spot.md) — fix that one first, then use this
ticket to add a targeted regression proving the admission gate itself is repaired, since
BUG-01's fix alone does not guarantee this specific consequence is covered by a test.

## Description of bug

`kairos_mtm.admission_check()` (wired into the day loop via E4-S10,
`strategy/kairos_papertrade.py`'s `_place_order_if_admitted`/`_place_batch_orders`) decides
whether to admit a new order by checking the CURRENT `DailySnapshot` — specifically
`last_snapshot`, the day loop's own snapshot, refreshed once per day from the same
fill/close-diffing logic described in BUG-01.

Because same-day fill+close positions are invisible to that diffing logic (BUG-01), they never
contribute to `initial_margin_used` / `gross_notional` in ANY `DailySnapshot` the admission
check ever sees. In an account whose trading activity is dominated by same-day round trips —
plausible and arguably common for tight-stop strategies on volatile daily-bar assets (this is
exactly the scenario in BUG-01's confirmed live repro: 3/3 real trades in that run were
same-day round trips) — `last_snapshot.initial_margin_used` stays at (or near) `0.0`
indefinitely. `admission_check`'s core test is:

```python
return (
    new_initial_margin_used <= new_equity * alloc_config.margin_utilization_cap
    and new_equity > 0.0
)
```

With `initial_margin_used` permanently near zero, this is satisfied for arbitrarily large new
orders — the leverage/exposure cap that E4-S10 exists specifically to enforce
(docs/tickets/E4-S10-admission-orders.md, docs/tickets/DESIGN_DOC_mtm_margin_leverage.md §4.3)
is defeated for exactly the trading pattern most likely to need it. This was not caught by
E5-S15's exposure-cap regression test (`tests/unit/test_kairos_papertrade_leverage_regression.py::test_exposure_cap_bounds_peak_gross_notional`)
because that test's synthetic candidates deliberately fill without triggering same-day SL/TP,
so the bug's precondition never arose.

## Definition of correct functionality

The admission gate's decision for a candidate order must be based on margin usage that
correctly accounts for ALL positions that existed — even briefly, same-day — since the last
admission check, not just positions that happen to still be `status='open'` at the moment a
`DailySnapshot` is taken. Concretely: in an account where every historical trade was a
same-day round trip, offering a new batch of orders whose aggregate notional would clearly
breach `margin_utilization_cap` if the prior trades' capital usage were correctly counted must
still result in some of that batch being `MARGIN_REJECTED` — the cap must be load-bearing
regardless of how quickly prior trades resolved.

## Reproduction instructions

Build on BUG-01's fix and its suggested test harness
(`tests/unit/test_kairos_papertrade_leverage_regression.py`'s `_FakeBarsProvider`/`_run_main`
pattern). Sketch:

1. Day 1: offer one or more candidates sized to consume most of `margin_utilization_cap`,
   with bars engineered so they fill AND hit SL/TP on that same day (same technique as
   BUG-01's repro: `High`/`Low` on the fill-day bar already crosses `target`/`stop`).
2. Day 2: offer a fresh batch of candidates whose aggregate notional, correctly margin-costed
   against what day 1's (already-closed) trades actually used, should be rejected in part —
   e.g. reuse E5-S15's `test_exposure_cap_bounds_peak_gross_notional` scenario structure but
   make wave 1 same-day round trips instead of trades that stay open into wave 2's check.
3. Before the BUG-01/BUG-02 fixes: `margin_rejected_count` for day 2's batch will be `0` (or
   lower than expected) even though it should be rejecting orders, because `last_snapshot`
   went into day 2 with `initial_margin_used == 0` despite day 1's real trading activity.
4. After the fixes: assert `margin_rejected_count > 0` for day 2's batch under a
   `margin_utilization_cap` deliberately set tight enough that day 1's (already-resolved)
   activity plus day 2's candidates would breach it if day 1 had been correctly counted.

## Context for whoever fixes this

- `strategy/kairos_papertrade.py`: `_place_order_if_admitted`, `_place_batch_orders`,
  `last_snapshot` assignment in the day loop (same region as BUG-01's diffing block).
- `strategy/kairos_mtm.py`: `admission_check` (output of E2-S04,
  docs/tickets/E2-S04-admission-check.md) — the pure function itself is almost certainly
  correct in isolation; the bug is entirely in what `DailySnapshot` main() feeds it.
- `docs/tickets/E4-S10-admission-orders.md` for the original admission-check ticket and its
  acceptance criteria.
- `docs/tickets/APPENDIX-A-standards.md` for style/testing/commit conventions.
