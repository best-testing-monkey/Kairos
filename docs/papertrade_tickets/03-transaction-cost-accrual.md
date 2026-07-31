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

- [ ] Fix the IBKR broker-profile commission schema mismatch (the `"tiered"`
  branch reads `tier.get("up_to")`/`tier.get("rate")`, keys that don't exist
  in the bundled `ibkr.json`) so simulated commission isn't silently zero.
  This makes the backtest *more* conservative, not less — fixing it won't
  explain the current loss, but leaving it unfixed means any future run
  understates real trading costs. **Lives in the external `phantom` package,
  not this repo.**
  - Note: for a genuine IBKR account, commissions matter a lot more for the
    account's tiny average trade size (~€18) than the 0.15% spread/slippage/fx
    does — worth validating against real IBKR crypto/CFD commission
    schedules before trusting this number is close to real-world.
- [ ] Consider a per-asset-class (or per-ticker-liquidity) `round_trip_cost_pct`
  in `AllocationConfig` instead of one flat 0.15%, so `NEG_EV_NET` gating is
  stricter for the illiquid alts that dominate the loss list.

## Files

- `phantom` package (external, broker-profile commission schema) — `ibkr.json`
- `strategy/allocation.py` (`AllocationConfig.round_trip_cost_pct`, `NEG_EV_NET` gate)
