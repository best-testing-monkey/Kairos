# E18-S04 — `MLBracketStrategy` wrapper

**Goal:** A new wrapper strategy, same shape as `ATRBracketStrategy`/`MetaLabelStrategy`, that
wraps any base strategy, evaluates the trained per-candidate classifiers (E18-S03) at inference
time, and overrides the base signal's `.stop`/`.target` with the EV-argmax candidate.

**Context:**
- Depends on E18-S03 (trained classifiers on disk) and E18-S02 (`extract_features()`).
- Wrapper pattern to copy: `ATRBracketStrategy` (`strategy/kairos_volatility.py:170-245` —
  constructor takes `base_strategy: Strategy`, recomputes a bracket, keeps tighter of
  {new, original}) and `MetaLabelStrategy` (`strategy/kairos_ml.py:183-234` — constructor takes
  `base_strategy`, calls `self.base_strategy.generate_signal(dist, current_price, history, context,
  **kwargs)` first, returns `None` unchanged if the base strategy returns `None`, otherwise builds
  a new `Signal` from the base one with specific fields overridden).
- Registration precedent: `strategy/kairos_orchestrator.py:717` (`atr_bracket =
  ATRBracketStrategy(base_strategy=TrendFollowingStrategy())`) and `:723` (`meta_label =
  MetaLabelStrategy(base_strategy=ExpectedValueStrategy())`) — add a similar line for
  `MLBracketStrategy`, but **do not add it to the active strategy registry list** in this story;
  construct it so it's importable/instantiable and covered by tests, leave wiring it into the live
  registry as a deliberate follow-up decision (this design's own non-goals section rules out live
  wiring before offline validation — E18-S05 — has run).
- `Signal` dataclass: `strategy/kairos_backtest.py:192-202` (`stop`/`target` are absolute prices;
  `metadata: dict` is the provenance field to use).
- **Train/serve feature parity is the main risk here.** E18-S03 saves a feature-order metadata file
  alongside each pickled classifier — this story's wrapper must load and use that exact feature
  order, not re-derive its own. Read `extract_features()`'s signature (E18-S02,
  `strategy/kairos_tpsl_features.py`) — the wrapper calls it exactly the same way training data was
  built, with the same feature set (price-history-only, no dist-derived features, per E18-S02's
  explicit v1 scope note) even though `dist` IS available live in `generate_signal()` — using it
  here anyway would silently mismatch what the model was trained on.

**Acceptance criteria:**
- [ ] `MLBracketStrategy(Strategy)` in `strategy/kairos_ml.py` (alongside `MetaLabelStrategy`/
  `GBMDirectionStrategy`), constructor takes `base_strategy: Strategy`, `model_dir: str` (defaults
  to `data/tpsl_models/`).
- [ ] `generate_signal()`: calls `base_strategy.generate_signal(...)` first; returns `None`
  unchanged if base returns `None`.
- [ ] If base returns a `Signal`: extracts features via `extract_features()` (E18-S02, using only
  the price-history feature set, matching the saved feature-order metadata), evaluates all loaded
  candidate classifiers' `predict_proba`, computes `EV = p_win * reward_pct - (1 - p_win) *
  risk_pct` per candidate (`risk_pct`/`reward_pct` derived from that candidate's stop/target vs.
  the base signal's `entry`, same formula shape as `allocation.py:147-148`), picks the argmax
  candidate.
- [ ] Overrides `.stop`/`.target` on a **new** `Signal` (don't mutate the base strategy's returned
  object — same non-mutation discipline `ATRBracketStrategy`/`MetaLabelStrategy` already follow).
- [ ] Sets `metadata["ml_bracket"] = {"original_stop": ..., "original_target": ..., "chosen_stop_pct":
  ..., "chosen_target_pct": ..., "predicted_p_win": ..., "predicted_ev": ...}` for auditability.
- [ ] If no trained classifiers are found at `model_dir` (e.g. E18-S03 hasn't been run), raises a
  clear error at construction time, not a silent pass-through — this is a wrapper that's
  meaningless without its trained models, unlike `ATRBracketStrategy` which has no such dependency.
- [ ] Unit test (add to `tests/unit/test_ml_strategies.py`, which already covers
  `strategy/kairos_ml.py`): mocked base strategy + mocked/fixture classifiers, verify override +
  metadata recorded correctly, and `None` passthrough when base strategy returns `None`.
- [ ] Unit test: feature vector passed to `predict_proba` matches the exact order in the saved
  feature-order metadata (regression test against train/serve skew).

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_ml.py`.
- [ ] New tests pass (`tests/unit/test_ml_strategies.py`); full suite green.
- [ ] Changes committed and `docs/todo.md` E18-S04 item checked off.
