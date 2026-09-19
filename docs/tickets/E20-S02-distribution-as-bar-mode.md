# E20-S02 — `distribution_as_bar` mode function + registration

**Goal:** Implement the mode function that appends E20-S01's synthetic bar to `history` and
returns a new `AssetPrediction` with `current_price`/`dist` left untouched, then register it in
`PREDICTION_USAGE_MODES` under the key `"distribution_as_bar"`.

**Context:**
- Depends on E20-S01 (`_build_synthetic_bar()`) and E19-S02 (`PREDICTION_USAGE_MODES` registry —
  find it wherever that story placed it, `kairos_orchestrator.py` or `kairos_meta.py`).
- Read `DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md` §2 for the exact function shape.
- Use `dataclasses.replace(pred, history=new_history)` — do not construct a new `AssetPrediction`
  by hand listing all 4 fields; `dataclasses.replace` guarantees `current_price`/`dist`/`symbol`
  stay exactly what they were on the input, which is the specific invariant this mode must
  preserve.
- **`current_price` must NOT change.** This is called out explicitly in the design doc as a
  deliberate decision, not an oversight — you can only trade at the real current price; the
  synthetic bar exists purely to feed `history`-based indicators.

**Acceptance criteria:**
- [ ] `distribution_as_bar(pred: AssetPrediction) -> AssetPrediction` in
  `strategy/kairos_prediction_usage.py`: builds the synthetic bar via `_build_synthetic_bar()`,
  returns `dataclasses.replace(pred, history=pd.concat([pred.history, bar.to_frame().T]))`.
- [ ] Registered in `PREDICTION_USAGE_MODES["distribution_as_bar"] = distribution_as_bar` (import
  into whichever module holds the registry if it's not the same file).
- [ ] Unit test: `distribution_as_bar(pred).current_price == pred.current_price` exactly.
- [ ] Unit test: `distribution_as_bar(pred).dist is pred.dist` (or `==`) — unchanged.
- [ ] Unit test: returned `history` has exactly `len(pred.history) + 1` rows; every row of
  `pred.history` is unchanged in the result (no in-place mutation — assert against a copy of the
  original taken before the call).
- [ ] Unit test: selecting `prediction_usage_mode="distribution_as_bar"` via
  `OrchestratorConfig` and running the E19-S02 dispatch produces the same result as calling
  `distribution_as_bar()` directly — confirms the registration wiring, not just the function in
  isolation.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_prediction_usage.py` (and the registry's
  file, if different).
- [ ] New tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E20-S02 item checked off.
