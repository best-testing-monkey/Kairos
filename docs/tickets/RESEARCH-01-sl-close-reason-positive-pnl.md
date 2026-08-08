# RESEARCH-01 — `close_reason='sl'` positions with positive realized_pnl (long, exit above entry)

**Type:** Research ticket. Goal: determine whether this is a real bug or explainable-but-surprising
behavior, and either (a) fix it if it's a real bug, with a reliable minimal reproduction and a
regression test, or (b) if it's correct behavior, document why in a code comment at the
relevant point in `phantom`-adjacent Kairos code (or in this ticket, closed out with findings)
and add a test that PINS the surprising-but-correct behavior so it doesn't get "fixed" away by
accident later.

## Observation

During the same live end-to-end run that surfaced BUG-01
(docs/tickets/BUG-01-same-day-fill-close-blind-spot.md — real Kronos inference, real price
data, `--max-leverage 2.0 --margin-utilization 0.6 --months-back 0.2 --capital 5000
--base-only`), all 3 closed positions were WLD-USD **long** trades tagged
`close_reason='sl'` (stop-loss) by phantom's engine:

```
ticker    direction  entry_price          exit_price           realized_pnl  close_reason
WLD-USD   long       0.3158000111579895   0.3449700495511293   40.94         sl
WLD-USD   long       0.3237000107765198   0.3449700495511293   23.96         sl
WLD-USD   long       0.31929999589920044  0.3449700495511293   33.76         sl
```

Two things look wrong at a glance:

1. **A stop-loss on a long position should trigger on a DOWNWARD move** (exit price below
   entry). Here `exit_price` (0.34497) is HIGHER than every `entry_price` (0.3158/0.3237/0.3193)
   — a nominally winning move — yet it's tagged `sl`, and `realized_pnl` is positive in all
   three cases.
2. **All three positions share the exact same `exit_price`** (0.3449700495511293, to full
   float precision) despite having three different `entry_price`s and (per BUG-01's findings)
   three different entry/exit dates (2026-08-03, 08-04, 08-05).

## Why this is a research ticket, not a bug ticket

Several explanations are plausible and need to be distinguished before deciding what "correct"
even means here:

- **(a)** An artifact of `_IntradayFallbackProvider`'s daily-bar evaluation combined with
  BUG-01's same-day-fill-and-close scenario — e.g. if phantom's engine evaluates SL/TP against
  the bar's `High`/`Low` but reports `close_reason` based on which threshold was crossed
  first in a way that's technically correct given the actual `stop_loss`/`take_profit` values
  attached to each order, even though the LABEL is surprising.
- **(b)** A genuine bug in `phantom_ledger` itself (third-party dependency, not this repo's
  code) — e.g. SL/TP direction/threshold mixed up somewhere in `SimulationEngine.run_backtest`
  or `PositionManager.close()`.
- **(c)** Something specific to how Kairos computes/passes `stop`/`target` into the `Order`
  object in `strategy/kairos_papertrade.py`'s day loop (e.g. `row.get("stop")`/`row.get("target")`
  from the allocation/signal pipeline being wrong-signed or mismatched for this particular
  strategy/candidate).
- **(d)** The identical `exit_price` across all three trades might indicate all three
  positions were actually evaluated/closed at the SAME wall-clock bar-fetch (e.g. a caching or
  "current price used for everything" bug), which would be a real bug regardless of the
  SL/TP-direction question.

## Task

1. Reproduce (ideally with the SAME live scenario first, to confirm it's not a one-off, then
   try to build a minimal deterministic/synthetic repro using the mocked `main()` harness
   already established in `tests/unit/test_kairos_papertrade_leverage_regression.py`).
2. Trace exactly what `stop_loss`/`take_profit` values were attached to each of the 3 orders —
   query the `orders`/`positions` tables' `stop_loss`/`take_profit` columns directly, and trace
   back to what `strategy/kairos_papertrade.py`'s day loop actually passed into the `Order(...)`
   constructor for each (`take_profit=row.get("target")`, `stop_loss=row.get("stop")` — verify
   these came from sane upstream values, not swapped or NaN).
3. Trace what bar(s) `_IntradayFallbackProvider.get_bars()` returned for each fill day, and how
   `phantom`'s `SimulationEngine.run_backtest()`/`PositionManager` (installed package, see
   `.venv/lib/python3.13/site-packages/phantom/engine/`) decide `close_reason` and `exit_price`
   from those bars vs. the position's `stop_loss`/`take_profit`.
4. Determine which of (a)-(d) above (or something else entirely) explains the observation.
5. **If a real bug is found** (in Kairos's own code — explanations (c) or a Kairos-side
   contributor to (d)): fix it, following this repo's normal ticket conventions (see
   `docs/tickets/APPENDIX-A-standards.md`) — write the "Definition of correct functionality"
   and "Reproduction instructions" yourself once the mechanism is understood (this research
   ticket deliberately doesn't prescribe them, since we don't yet know what "correct" means
   here), then fix + add a regression test pinning the corrected behavior.
6. **If it's explainable/correct behavior** (e.g. explanation (a), or a confirmed
   `phantom_ledger` third-party bug outside this repo's fix scope): add a short comment at the
   relevant Kairos-side call site explaining why a "sl"-tagged position can show a nominal win
   and identical exit prices are possible, and add a test that pins this specific
   surprising-but-correct scenario so a future refactor doesn't "fix" it into actually-wrong
   behavior. If it's a confirmed third-party `phantom_ledger` bug outside this repo, document
   that clearly (what, where, why not fixable here) rather than attempting to patch a vendored
   dependency.
7. Report findings either way — do not leave this ambiguous. "Investigated, inconclusive" is
   not an acceptable end state; if truly stuck after a solid effort, report EXACTLY what was
   ruled out and what remains uncertain, per this repo's "characterization not speculation"
   testing philosophy (see `tests/unit/test_kairos_papertrade_loss_repro.py`'s module docstring
   for the house style of honest, evidence-based reporting).

## Context

- `strategy/kairos_papertrade.py`: the day loop's `Order(...)` construction (~line 1830-ish,
  search for `take_profit=row.get("target")`), `_IntradayFallbackProvider`.
- `phantom` (installed package): `phantom/engine/simulation_engine.py`,
  `phantom/engine/position_manager.py`, `phantom/engine/order_manager.py` — read these to
  understand the real SL/TP evaluation and `close_reason` assignment logic.
- The live run's artifacts (if still present in scratch) or a fresh repro run following the
  same recipe as BUG-01's ticket.
