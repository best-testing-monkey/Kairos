# E19-S02 — `PREDICTION_USAGE_MODES` registry + `_run_day()` hook

**Goal:** Add a module-level registry mapping mode name → transform function, and wire a single
insertion point in `_run_day()` that applies the selected mode's transform to every
`AssetPrediction` before strategies run. Only the identity (`"last_real_bar"`) entry exists after
this story — no behavior change yet.

**Context:**
- Depends on E19-S01 (`prediction_usage_mode` field).
- Read `docs/tickets/DESIGN_DOC_prediction_usage_modes_architecture.md` §3 for the exact code
  shape to build.
- `AssetPrediction` — `strategy/kairos_meta.py:116-122`, exactly 4 fields: `symbol`, `dist`,
  `current_price`, `history`.
- `_run_day()` — `strategy/kairos_orchestrator.py`, the branch that builds `multi_preds` is at
  ~line 1093-1097:
  ```python
  if self.config.no_prediction:
      multi_preds = self._make_realized_predictions(date, histories, naive=self.config.naive_baseline)
  else:
      multi_preds = self.multi_predictor.predict_all(histories)
  ```
  Add the registry dispatch **immediately after** this block, before context enrichment / the
  per-strategy loop (~line 1099 onward) — do not restructure the existing if/else, just append.
- Use `dataclasses.replace()` (stdlib) for the identity function, not a hand-written field copy —
  `AssetPrediction` is a plain `@dataclass`, `dataclasses.replace(pred)` with no kwargs returns an
  equal copy; this is the simplest correct identity transform and doubles as a smoke test that the
  hook itself doesn't mutate anything.

**Acceptance criteria:**
- [ ] Module-level registry in `kairos_orchestrator.py` (or `kairos_meta.py`, wherever
  `AssetPrediction` already lives — implementer's call, keep it next to the dataclass it operates
  on):
  ```python
  PredictionUsageFn = Callable[[AssetPrediction], AssetPrediction]
  PREDICTION_USAGE_MODES: Dict[str, PredictionUsageFn] = {
      "last_real_bar": lambda pred: pred,
  }
  ```
- [ ] `_run_day()` applies the dispatch right after `multi_preds` is built:
  ```python
  usage_fn = PREDICTION_USAGE_MODES[self.config.prediction_usage_mode]
  multi_preds = {sym: usage_fn(pred) for sym, pred in multi_preds.items()}
  ```
- [ ] Unknown `prediction_usage_mode` string raises a clear `KeyError` (the dict lookup above
  already does this — do not wrap it in a `try/except` that swallows or defaults silently).
- [ ] Unit test: register a dummy mode function in a test-local copy/monkeypatch of the registry,
  select it via `OrchestratorConfig(prediction_usage_mode=...)`, confirm `_run_day()` calls it
  (assert on a side effect or a distinguishing return value).
- [ ] Unit test: default config (`prediction_usage_mode="last_real_bar"`) leaves `multi_preds`
  unchanged — same `dist`/`history`/`current_price` values as before the dispatch was added.
- [ ] Unit test: an invalid mode string raises `KeyError` at the dispatch point, not a silent no-op.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_orchestrator.py` (and `kairos_meta.py` if
  the registry lands there instead).
- [ ] New tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E19-S02 item checked off.
