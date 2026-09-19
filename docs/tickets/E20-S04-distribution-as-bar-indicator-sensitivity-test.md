# E20-S04 — Indicator sensitivity + no-effect-on-non-history-strategies tests

**Goal:** Confirm `distribution_as_bar` actually changes indicator-based strategies' behavior
(RSI/MACD see the forecast) and confirms it has **zero effect** on strategies that don't read
`history` at all — the scoping claim in
`DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md` §3.

**Context:**
- Depends on E20-S02.
- Indicator-based strategies to test:
  - `RSIFilterStrategy` (`strategy/kairos_backtest.py`, class ~line 1136; RSI calc via
    `self._rsi(history)` reading `history["close"].values[-self.period:]`, ~line 1147-1148;
    direction/gate logic ~line 1161-1171).
  - `MACDFilterStrategy` (`strategy/kairos_backtest.py`, class ~line 1193; MACD via pandas `.ewm()`
    on `history`, ~line 1204-1211; gate logic ~line 1213-1223).
- Non-history strategy to test: `TrendFollowingStrategy` (`strategy/kairos_backtest.py:783-802`) —
  direction is `sign(dist.stats["close"]["mean"] - current_price)` only, never reads `history`.
- Test shape: build one `AssetPrediction` fixture, call each strategy's `generate_signal()` twice —
  once with `pred.history` as-is, once with `distribution_as_bar(pred).history` — same `dist`/
  `current_price` both times.

**Acceptance criteria:**
- [ ] Unit test: `RSIFilterStrategy.generate_signal()` produces a different result (different
  `Signal`, or `None` vs. non-`None`, or a materially different RSI-derived gate outcome) between
  the two `history` variants, for at least one crafted fixture where the synthetic bar's close
  crosses an RSI threshold.
- [ ] Unit test: same for `MACDFilterStrategy` with a fixture crossing a MACD threshold.
- [ ] Unit test: `TrendFollowingStrategy.generate_signal()` produces **byte-identical** `Signal`
  output (or identical `None`) between the two `history` variants, for the same `dist`/
  `current_price` — proves the mode has no effect on strategies that don't read `history` for
  direction.
- [ ] Test file location: extend `tests/unit/test_prediction_usage.py` (from E20-S01/E20-S02) or
  add `tests/unit/test_strategy_signals.py` cases if that's a better fit for strategy-level
  assertions — implementer's call.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to the changed test file(s).
- [ ] New tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E20-S04 item checked off.
