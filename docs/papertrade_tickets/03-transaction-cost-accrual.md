# Factor 3: Transaction cost accrual

Source: `docs/papertrade_loss_analysis.md` §4, Factor 3

## Problem

Measured realized round-trip cost across all 539 trades in the recorded run
was **exactly 0.15% of notional** (spread €2.90 + slippage €1.93 + fx €9.67 +
commission €0.00, over €9,665 total notional) — matching
`strategy/allocation.py`'s `round_trip_cost_pct=0.15` assumption almost
exactly. This rules out "the cost assumption is too optimistic" as the main
driver of the recorded loss.

Two real issues remain:
1. Commission is silently always €0.00 — a schema mismatch between
   `phantom`'s `"tiered"` commission model and the bundled IBKR
   broker-profile JSON's tier keys (external to Kairos, lives in the
   `phantom` package).
2. 0.15% is a single constant applied uniformly, even though the top losing
   tickers (`LDO-USD`, `AAVE-USD`, `ATOM-USD`, `XTZ-USD`, `AXS-USD`) are
   lower-liquidity alts likely to have wider real-world spreads than majors.

## Statistic to optimize

Per-asset-class realized cost % (not just the blended average), and whether
`min_ev_pct`/`round_trip_cost_pct` should vary by liquidity/asset-class
rather than being one global constant.

## Concrete changes

- [x] Fix the IBKR broker-profile commission schema mismatch. Root-caused
  2026-09-02 via a live IB Gateway probe (kairos-07 session): the bundled
  `ibkr.json` itself is fine (it's `"per_share"`, not `"tiered"`, and has
  been since before this ticket was opened) — the actual bug was that the
  **live `data/phantom_ledger/phantom.db` `broker_profiles` row was stale**,
  still on an old `"tiered"` schema whose tier dicts used
  `"min_volume"`/`"max_volume"`/`"per_share"` keys instead of the
  `"up_to"`/`"rate"` keys `CommissionModel.calculate()`'s tiered branch
  reads — so every tier's rate defaulted to 0.0, silently pricing every
  simulated trade at €0.00 commission. Root cause of the non-migration:
  `_ensure_broker_profile()` (`strategy/kairos_papertrade.py`) returned as
  soon as ANY profile named "IBKR" existed, so a `phantom_data_dir` created
  before this schema was ever correct stayed on the stale config forever.
  Fixed in two places:
  - Kairos-local: `_ensure_broker_profile()` now diffs the stored profile
    against the bundled JSON and calls `client.brokers.update()` to heal it
    when they differ (idempotent, same pattern as `_sync_margin_classes()`
    right after it) — this is the fix that actually matters for Kairos's
    live commission bug, independent of any phantom_ledger version.
  - `phantom_ledger` (external, committed locally as `3bde7a3`, not yet
    pushed/lockfile-bumped): `CommissionModel.calculate()`'s tiered branch
    now raises `ValueError` on a tier missing `"up_to"`/`"rate"` instead of
    silently defaulting to a 0.0 rate, so a future schema mismatch fails
    loudly instead of repeating this bug undetected.
  - Note: for a genuine IBKR account, commissions matter a lot more for the
    account's tiny average trade size (~€18) than the 0.15% spread/slippage/fx
    does — see the next item, which is where this bites.
- [ ] Consider a per-asset-class (or per-ticker-liquidity) `round_trip_cost_pct`
  in `AllocationConfig` instead of one flat 0.15%, so `NEG_EV_NET` gating is
  stricter for the illiquid alts that dominate the loss list. **Still open —
  new data below, but the default itself was deliberately NOT changed** (see
  `AllocationConfig`'s own docstring: "defaults are deliberately round
  numbers swept in Phantom Ledger. Do not ship precise-looking fitted
  values." — a single spot-check isn't a sweep, and `round_trip_cost_pct`
  is a documented RFC §3.1 default, not a constant to silently overwrite).
  - **New data (2026-09-02, live IB Gateway paper account, whatIfOrder)**:
    IBKR's US stock commission floor is $1.00/order regardless of size, so
    round-trip cost is dominated by a ~$2 flat floor at small notional, not
    a percentage:
    | notional | measured round-trip cost |
    |---|---|
    | $277 | 0.72% |
    | $326 | 0.61% |
    | $652 | 0.31% |
    | $1,630 | 0.12% |
    | $3,378 | 0.06% |
    | $65,212 | 0.00% |
    All six points match `$2 / notional` almost exactly — the existing
    0.15% constant is only accurate near ~$1,300 notional. Kairos's measured
    average trade (~€18, see §"Statistic to optimize" above) is far below
    that pivot and even below IBKR's fractional-share API barrier entirely
    (error 10243) — at that size, real round-trip cost is closer to
    `$2/€18 ≈ 11%`, not 0.15%. `compute_derived()` in `strategy/allocation.py`
    computes `ev_net`/`score`/`kelly_frac` (hence which candidates get
    selected at all) BEFORE `size_selected()` sizes a real position, so a
    correct fix needs either a conservative pre-sizing assumption or a
    real post-sizing veto once `config.equity`-derived notional is known —
    not a single swept constant. Needs a real methodology decision (and
    probably a proper sweep, per the docstring above), not a spot patch.

## Files

- `phantom` package (external, broker-profile commission schema) — `ibkr.json`,
  `src/phantom/models/broker.py` (`CommissionModel.calculate()`)
- `strategy/kairos_papertrade.py` (`_ensure_broker_profile()`)
- `strategy/allocation.py` (`AllocationConfig.round_trip_cost_pct`, `NEG_EV_NET` gate)
