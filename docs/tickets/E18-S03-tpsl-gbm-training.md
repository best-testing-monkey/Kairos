# E18-S03 — Train per-candidate GBM classifiers with purged CV

**Goal:** Join E18-S01's labels with E18-S02's features and train one `GradientBoostedStumps`
classifier per grid candidate, predicting `P(target hit before stop)`, using a purged/embargoed
time-based split (not a random split). Check per-asset-class signal counts before deciding
pooled-vs-per-class, and save the trained classifiers to disk.

**Context:**
- Depends on E18-S01 (`tpsl_label_candidates` table) and E18-S02 (`strategy/kairos_tpsl_features.py`).
- Read `DESIGN_DOC_ml_tpsl_optimization.md` §4 (2026-09-19 revision) — **do not use LightGBM or any
  new dependency.** `strategy/kairos_ml.py:373-430`'s `GradientBoostedStumps` is the model: numpy
  only, `fit(X: np.ndarray, y: np.ndarray)` (binary `{0,1}` labels), `predict_proba(X) ->
  np.ndarray`. Import it from `kairos_ml.py`, don't copy/reimplement it.
- One classifier instance per `(stop_pct, target_pct)` grid combination (9 total, per E18-S01's
  grid) — `y` for each is that candidate's `hit_target_first` column from `tpsl_label_candidates`
  (rows with `resolved=0`/`NULL` excluded from that candidate's training set entirely, not imputed).
- **Purged/embargoed split**: sort signals by `as_of`, hold out the most recent N% (e.g. last 20%)
  as validation, and drop training rows within an embargo gap (e.g. 5 bars' worth of time, exact
  value TBD/configurable) immediately before the validation cutoff — per
  `DESIGN_DOC_ml_tpsl_optimization.md` §3's explicit citation of AFML's warning that a naive random
  split leaks adjacent-in-time volatility-regime information for this exact parameter class. This
  is the one part of this story that needs care, not mechanical copying — if anything about the
  embargo window size is unclear, pick a documented, reasonable default and flag it in a comment
  rather than guessing silently.
- **Per-class data-sufficiency check**: `CLASS_STATS_MIN_SIGNALS = 30`
  (`strategy/kairos_pipeline.py:1359`) is this repo's existing threshold for "enough signals to
  trust a class-level statistic" (see `kairos_pipeline.strategy_class_stats()`'s fallback logic for
  the precedent). Count resolved labeled signals per `asset_class_for()` bucket; if any bucket
  (or the whole training set) is below this threshold, log a clear warning and fall back to a
  single pooled-across-classes model rather than a fragile per-class one. Per this repo's own
  tracked project state, the corpus is known to skew heavily equity — expect this warning to fire
  for crypto/fx in practice; that's a correct result of the check, not a bug to "fix" by lowering
  the threshold.
- New script: `scripts/train_tpsl_model.py` (mirrors `scripts/run_oracle_dedup.py`'s /
  `scripts/backfill_class_stats.py`'s placement as a standalone `pipeline_results.db`-reading
  tool).
- Persist trained classifiers via stdlib `pickle` (no new serialization dependency) to e.g.
  `data/tpsl_models/<stop_pct>_<target_pct>.pkl`, plus a small metadata JSON recording the feature
  list/order used (E18-S04's wrapper must feed features in this exact same order — mismatched
  feature order/set between training and inference is a real, silent-failure risk, call it out in
  a code comment at both the save and load sites).

**Acceptance criteria:**
- [ ] Script joins `tpsl_label_candidates` (E18-S01) with features computed via
  `extract_features()` (E18-S02) for each signal (re-fetching `history` per signal the same way
  E18-S02's docstring describes).
- [ ] Trains 9 `GradientBoostedStumps` instances (one per grid candidate), each via purged
  time-based split.
- [ ] Per-class signal count check implemented and logged; falls back to pooled training when any
  class (or the whole set) is below `CLASS_STATS_MIN_SIGNALS`.
- [ ] Saves each trained classifier + a feature-order metadata file to `data/tpsl_models/`.
- [ ] Unit test: purged split function, given a synthetic sorted-by-time dataset and an embargo
  window, produces train/validation sets with the documented time gap between them (no
  adjacent-timestamp leakage across the split boundary).
- [ ] Unit test: per-class sufficiency check correctly falls back to pooled training on a synthetic
  dataset where one class has fewer than `CLASS_STATS_MIN_SIGNALS` rows.
- [ ] Unit test: saved classifier round-trips through pickle and produces identical
  `predict_proba()` output before/after save+load, on a small fixture.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `scripts/train_tpsl_model.py`.
- [ ] New tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E18-S03 item checked off. Do not commit
  `data/tpsl_models/` output artifacts (per `APPENDIX-A-standards.md`'s "do not commit generated
  outputs" rule).
