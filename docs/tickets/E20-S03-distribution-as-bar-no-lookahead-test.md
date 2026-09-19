# E20-S03 — No-lookahead test for `distribution_as_bar`

**Goal:** A dedicated test proving `_build_synthetic_bar()`/`distribution_as_bar()` read nothing
beyond the `AssetPrediction` passed in — no reaching into a full/future price DataFrame, no
module-level state. This must pass before the mode is considered safe to enable anywhere, per
`DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md` §4's explicit caution: this project has
hit this exact bug shape twice before (the naive-baseline zero-drift trap, and the oracle-decision
peek bug — both documented in this repo's CLAUDE.md, 2026-08-28/2026-09-01).

**Context:**
- Depends on E20-S01/E20-S02.
- Read `tests/unit/test_naive_no_lookahead.py` in full — this repo already has an established
  pattern for exactly this class of test (for the naive-baseline mode). Mirror its structure/style
  rather than inventing a new one from scratch.
- The claim to verify, precisely: `distribution_as_bar()`'s output is a pure function of its single
  `pred: AssetPrediction` argument. It must not import or reference any module-level "full history"
  object, any other symbol's data, or any date later than what's already encoded in `pred` itself
  (`pred.dist` was already built from whatever forecast basis the evaluation mode — oracle/naive/
  live model — chose; this function doesn't get to see anything beyond that).

**Acceptance criteria:**
- [ ] Test constructs an `AssetPrediction` fixture that is deliberately the ONLY data available
  (e.g. no other module-level DataFrame in scope, or a mock/spy object in place of anything that
  could be a full-history reference) and confirms `distribution_as_bar()` still produces a correct
  result using only `pred.dist`/`pred.history`/`pred.current_price`.
- [ ] Test asserts, via `inspect.getclosurevars()` or an equivalent static check, that
  `_build_synthetic_bar`/`distribution_as_bar` have no closure-captured references to anything
  other than stdlib/pandas/numpy names and their own parameters — catches an accidental future
  regression where someone adds a "helpful" reference to outside state.
- [ ] Test explicitly covers the oracle-mode case: an `AssetPrediction` whose `.dist` was built via
  `KairosDistribution.from_bar()` from a real future-peeked bar (`_make_realized_predictions`,
  `kairos_orchestrator.py:964-1034`) — confirm `distribution_as_bar()` only touches `.dist`/
  `.history`/`.current_price` on that `AssetPrediction`, the same as for a live-model prediction.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to the new test file.
- [ ] New test passes; full suite green.
- [ ] Changes committed and `docs/todo.md` E20-S03 item checked off.
