# E21-S05 — Fix `_build_synthetic_bar()`'s OHLC-consistency gap

**Goal:** `_build_synthetic_bar()` (`strategy/kairos_prediction_usage.py`, ~line 97) takes
open/high/low/close independently from each column's own marginal `pct_50`, with no
OHLC-consistency check — found by code review, can produce a bar with `high < low` or `close`
outside `[low, high]`.

**Context:**
- Each of high/low/close is `KairosDistribution.stats[col]["pct_50"]`
  (`strategy/kairos_backtest.py`'s `_compute_stats()`), computed as an **independent per-column
  percentile**, not a paired/joint sample from one simulated path. Given this project's
  documented high-kurtosis sample noise (see CLAUDE.md's "Kurtosis filter threshold" section —
  discrete token sampling routinely produces excess kurtosis well above normal), this can yield a
  synthetic bar where `high < low`, or `close` falls outside `[low, high]`.
- Consequence: any strategy computing ATR/Donchian/range-based sizing off that bar gets a
  negative or nonsensical range. `tests/unit/test_prediction_usage*.py`'s existing fixtures all
  use already-consistent high>close>low values, so this was never exercised.
- Fix: after computing the three independent percentiles, reconcile them into a valid OHLC bar —
  e.g. `high = max(open, high, low, close)`, `low = min(open, high, low, close)`, keeping `close`
  as computed (clamped into `[low, high]` if it still falls outside after the max/min pass).
  Document the reconciliation approach in a code comment — this is a deliberate approximation
  (the three columns aren't jointly sampled), not a subtle bug, so a future reader should
  understand why the clamp exists.

**Acceptance criteria:**
- [ ] `_build_synthetic_bar()` always returns a bar satisfying `low <= open, close <= high` and
  `low <= high` — verified by construction, not by luck of typical input values.
- [ ] Unit test: adversarial `dist.stats` values engineered so the independent per-column
  percentiles would otherwise violate the invariant (e.g. `high.pct_50 < low.pct_50`) — assert
  the returned bar is still internally consistent after the fix.
- [ ] Unit test: existing well-behaved fixtures (already-consistent percentiles) are unaffected —
  the fix must be a no-op when the input is already consistent, not silently altering good data.
- [ ] All existing `test_prediction_usage*.py` tests still pass unchanged.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_prediction_usage.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Commit with "E21-S05" in the subject line. No Co-Authored-By trailer.
- [ ] Do NOT edit `docs/todo.md`.
