# E19-S03 — Default-mode parity regression test

**Goal:** Prove that `_run_day()` with the default `prediction_usage_mode="last_real_bar"`
produces byte-identical `Signal` output to pre-E19 code. This is the load-bearing guarantee of the
whole architecture change — per `DESIGN_DOC_prediction_usage_modes_architecture.md` §5, the
registry wiring must not accidentally alter `dist`/`history`/`current_price` even under the
default mode.

**Context:**
- Depends on E19-S02 (registry + `_run_day()` hook).
- Search `tests/unit/` for existing `_run_day()`/`KairosOrchestrator` fixtures before writing new
  ones — `tests/unit/test_orchestrator_allocator.py` and `tests/unit/test_realized_predictions.py`
  both already exercise `_run_day()`-adjacent code; reuse whichever fixture setup is closest rather
  than inventing a third one. If neither fits, a new minimal fixture is fine — but check first.
- "Byte-identical" here means: same `Signal.direction`/`.entry`/`.stop`/`.target`/`.size`/
  `.expected_value` for every strategy/symbol/date in the fixture, comparing a run with the E19
  changes applied against a run of the same fixture on the pre-E19 code path (i.e., before this
  story existed the default behavior WAS the current behavior — assert current output matches
  itself under the new plumbing, not a stored golden file from a different commit).

**Acceptance criteria:**
- [ ] Test constructs (or reuses) a small multi-symbol, multi-date fixture and runs
  `KairosOrchestrator.run_backtest()`/`_run_day()` twice: once with default `OrchestratorConfig()`
  (mode defaults to `"last_real_bar"`), once with `OrchestratorConfig()` from before E19-S02 landed
  — since that's not literally re-runnable, instead assert the **structural** invariant directly:
  for every `AssetPrediction` in `multi_preds` after the E19-S02 dispatch, `dist is` the same
  object (or `==`) and `history`/`current_price` are unchanged from what
  `multi_predictor.predict_all()`/`_make_realized_predictions()` produced before the dispatch was
  applied. This is the precise, mechanically-checkable version of "parity."
- [ ] Test covers both branches of the pre-existing `no_prediction`/`naive_baseline` if/else (§3 of
  the architecture doc's "composability" table) — confirms the new hook doesn't only work for one
  of the two evaluation-basis branches.
- [ ] Full existing test suite still green — this is itself part of the parity guarantee; any
  pre-existing test whose expected `Signal` values changed would indicate a real regression.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to the new/changed test file.
- [ ] New test passes; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Changes committed and `docs/todo.md` E19-S03 item checked off.
