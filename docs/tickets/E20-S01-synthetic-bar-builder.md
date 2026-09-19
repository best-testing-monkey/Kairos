# E20-S01 — `_build_synthetic_bar()` construction function

**Goal:** A pure function that builds one synthetic OHLCV bar from a `KairosDistribution`'s
percentile stats plus the current price, per
`DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md` §2. No wiring into the mode registry
yet — that's E20-S02.

**Context:**
- Depends on E19-S01/E19-S02 existing (this epic builds the second registry entry) but this
  specific story has no code dependency on them — it's a standalone pure function, testable in
  isolation.
- `KairosDistribution.stats` — `strategy/kairos_backtest.py:286-341` (`_compute_stats()`). Keys are
  `"open"`/`"high"`/`"low"`/`"close"` only — **no `"volume"` key** (confirmed by reading the
  method: it loops `for col in ["open", "high", "low", "close"]`). Each has `"pct_50"` among its
  percentile keys.
- `AssetPrediction` — `strategy/kairos_meta.py:116-122`.
- Volume: **carry the last real bar's volume forward unchanged** — do not attempt to compute a
  volume percentile from `dist.df["volume"]` in this story (that raw column exists but is
  unaggregated; per the design doc §2 this is an explicit, deliberate v1 simplification, not a
  gap to fill here).
- Timestamp: the synthetic bar's index must be `pred.history.index[-1]` advanced by one interval
  step. **Search for an existing interval-to-timedelta/offset helper before writing new date
  math** — check `strategy/kairos_signals.py` and `strategy/kairos_papertrade.py` for anything
  already converting an interval string (`"1h"`/`"4h"`/`"1d"`) to a `pd.Timedelta`/offset (the
  multi-interval rollout, `DESIGN_DOC_multi_interval_1h.md`, likely needed exactly this — check
  there first). If nothing reusable exists, a small local `_advance_one_interval(ts, interval)`
  using `pd.Timedelta` is fine, but check first per this project's reuse-before-writing convention.
- New module: `strategy/kairos_prediction_usage.py` (new file — this is a new architectural axis,
  doesn't belong bolted onto an existing strategy file).

**Acceptance criteria:**
- [ ] `_build_synthetic_bar(pred: AssetPrediction, interval: str) -> pd.Series` implemented:
  `open=pred.current_price`, `high=pred.dist.stats["high"]["pct_50"]`,
  `low=pred.dist.stats["low"]["pct_50"]`, `close=pred.dist.stats["close"]["pct_50"]`,
  `volume=pred.history.iloc[-1]["volume"]`, indexed/named by `pred.history.index[-1]` advanced one
  `interval` step.
- [ ] Unit test: fixture `AssetPrediction` with known `dist.stats` values — returned bar's
  `open`/`high`/`low`/`close` match expected values exactly.
- [ ] Unit test: returned bar's `volume` equals `pred.history.iloc[-1]["volume"]` exactly (carry-
  forward, not recomputed).
- [ ] Unit test: returned bar's index is exactly one interval step after `pred.history.index[-1]`,
  for at least two different `interval` values (e.g. `"1h"` and `"1d"`).

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_prediction_usage.py`.
- [ ] New tests pass (`tests/unit/test_prediction_usage.py`); full suite green.
- [ ] Changes committed and `docs/todo.md` E20-S01 item checked off.
