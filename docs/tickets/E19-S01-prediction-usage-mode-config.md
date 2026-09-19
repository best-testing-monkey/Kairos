# E19-S01 — Add `prediction_usage_mode` to `OrchestratorConfig`

**Goal:** Add a new config field, `prediction_usage_mode: str = "last_real_bar"`, to
`OrchestratorConfig` — pure plumbing, zero behavior change. This is the config surface E19-S02's
registry dispatch and E19-S03's parity test build on.

**Context:**
- Read `docs/tickets/DESIGN_DOC_prediction_usage_modes_architecture.md` §3 for the intent.
- `strategy/kairos_orchestrator.py`, `class OrchestratorConfig` — a plain `@dataclass`. Existing
  mode-like fields for reference: `no_prediction: bool = False` (~line 402), `naive_baseline: bool
  = False` (~line 412). Add the new field in the same area of the dataclass, near those two.
- Do not touch `_run_day()`'s dispatch logic in this story — that's E19-S02. This story only adds
  the field and its default.

**Acceptance criteria:**
- [ ] `prediction_usage_mode: str = "last_real_bar"` added to `OrchestratorConfig`.
- [ ] Unit test (add to `tests/unit/test_orchestrator_config.py`, which already covers
  `OrchestratorConfig`): `OrchestratorConfig().prediction_usage_mode == "last_real_bar"`.
- [ ] Unit test: `OrchestratorConfig(prediction_usage_mode="distribution_as_bar").prediction_usage_mode
  == "distribution_as_bar"` — confirms the field is a normal constructor kwarg, no special-casing
  needed yet (E19-S02 handles validating it's a known registered mode).
- [ ] No other file changes — this story adds a field with no consumer yet.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_orchestrator.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`) —
  confirm no existing test asserting `OrchestratorConfig()`'s full field set breaks (grep for
  `OrchestratorConfig(` in `tests/unit/` first if unsure).
- [ ] Changes committed and `docs/todo.md` E19-S01 item checked off.
