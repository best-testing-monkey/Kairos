# Factor 4: Order execution mechanics

Source: `docs/papertrade_loss_analysis.md` §4, Factor 4

## Problem

Every order fills at the next real bar's **Open** price
(`phantom`'s `OrderManager.evaluate`), one report-cycle after the signal was
generated (the "ONE-REPORT LAG" design documented in
`kairos_papertrade.py`'s module docstring). But `take_profit`/`stop_loss` on
the placed `Order` (`kairos_papertrade.py:527-528`) are copied **verbatim**
from the signal's `target`/`stop`, which the originating strategy computed
relative to the **stale, report-time entry price** — not the actual fill
price.

When a same-bar tie occurs between hitting stop and target, `phantom`'s
default conflict resolution (`mode="conservative"`, never overridden anywhere
in this call chain) always resolves to **SL wins**. Measured result: of 539
closes, **409 hit stop-loss** vs. only **120 hit take-profit** — a ~3.4:1
ratio.

## Statistic to optimize

The SL:TP hit-rate ratio (currently ~3.4:1) and the realized risk/reward
ratio actually achieved vs. what each strategy's own distribution-derived
stop/target implied. Also: the size of the overnight gap between report-time
signal `entry` and the actual next-bar-open fill price.

## Concrete changes

- [ ] Re-base `stop`/`target` off the **actual fill price** at order-creation
  time (`kairos_papertrade.py:523-529`) instead of copying the signal-time
  values verbatim — e.g. recompute `stop`/`target` as the same
  %-distance-from-entry the strategy originally intended, applied to the
  real fill price.
- [ ] Evaluate whether `phantom`'s tie-resolution mode should be overridden
  (it currently always favors SL on same-bar hits, structurally suppressing
  the payoff side of a low-win-rate/high-payoff strategy).
- [ ] Consider limit orders (bounded slippage on entry) instead of market
  orders for signals where the report-time entry and typical next-bar gap
  size make market fills risky.

## Files

- `strategy/kairos_papertrade.py` (order construction, lines ~523-529)
- `phantom` package (`OrderManager.evaluate`, tie-resolution mode)
