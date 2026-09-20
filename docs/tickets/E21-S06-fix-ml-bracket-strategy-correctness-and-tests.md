# E21-S06 — Fix `MLBracketStrategy` correctness bugs + add real test coverage

**Goal:** Code review found 7 distinct correctness bugs in `MLBracketStrategy`
(`strategy/kairos_ml.py`) plus a test-coverage gap that let all 7 ship with green tests. Fix all
of them together in one pass — they're small, tightly coupled within one ~150-line class, and
several interact (fixing the context-key bug changes what reaches the FLAT-direction and
asset-class-filter code paths). This is the largest/most careful story in this batch; take the
time to read the whole class before editing any of it, per this project's stated ladder ("read
fully, then be lazy").

**Context — read all 7 findings below in full, then read current `strategy/kairos_ml.py` source
in full (line numbers below are from the review, verify against current source since they may
have shifted):**

1. **(~line 1269) Missing context keys — silently never fires once live.**
   `MLBracketStrategy` requires `context["ticker"]`/`context["interval"]`, but no real context
   builder in the codebase ever sets those keys. `kairos_orchestrator.py`'s `_run_day` context
   dict uses `current_symbol`, not `ticker`; `kairos_signals.py`'s `_build_context` never sets
   `interval` at all. Once registered in the live strategy list, `context.get("ticker")` is
   always `None`, so every call takes the "missing context" branch and returns `base_sig`
   unchanged — silently, no error. Fix: read `context.get("current_symbol")` for the ticker
   (matching what real context builders actually set — verify this against current source, don't
   assume). For interval, since no context builder sets it at all, the more robust fix is
   accepting `interval` as a **constructor parameter** to `MLBracketStrategy.__init__` (it's
   fixed per orchestrator run anyway) rather than expecting it from a context key nothing
   provides — but verify no cleaner path exists first.

2. **(~line 1276) Self-referential features under `distribution_as_bar` mode.**
   `as_of_date = history.index[-1]` — under `prediction_usage_mode="distribution_as_bar"`, the
   last row of `history` IS the synthetic forecast-derived bar (see `E20-S02`), so
   `extract_features()`'s no-lookahead promise (its own docstring: output unchanged whether
   history is truncated exactly at `as_of` or has extra rows after) becomes a no-op — ATR/
   realized_vol/trend/range-position get computed including the fabricated bar as real history,
   making the classifier's features partly self-referential on the model's own forecast. Fix
   mechanism is your call — e.g. have `MLBracketStrategy` exclude the last history row before
   calling `extract_features()` when it can detect a synthetic bar was appended, or find another
   robust way to guarantee it never computes features from a bar that hasn't really happened.
   Document whichever approach you take and why in a code comment — this is exactly the bug
   class (lookahead via a synthetic/forecast bar) this project has been bitten by twice before
   (see CLAUDE.md's naive-baseline/oracle-peek sections), so be rigorous, not quick.

3. **(~line 1308) `Direction.FLAT` mishandled as SHORT.**
   `if base_sig.direction == Direction.LONG: ... else: # SHORT` — several strategies
   (`kairos_crypto.py`, `kairos_forex.py`, `kairos_meta.py`, `kairos_path.py`) return explicit
   `Signal(direction=Direction.FLAT, ...)` rather than `None`. Wrapping one of those,
   `MLBracketStrategy` computes `candidate_stop`/`candidate_target` using the SHORT formula
   against a non-directional signal, stamping nonsensical values into the returned `Signal`'s
   metadata even though `direction` stays FLAT. Fix: explicit branch — for any direction other
   than LONG/SHORT, skip ML bracket selection and return `base_sig` unchanged (same as the
   missing-context fallback path).

4. **(~line 1301) No asset-class filtering on the argmax-EV loop.**
   `scripts/train_tpsl_model.py` trains per-class models keyed `f"{stop_pct}_{target_pct}_{cls}"`
   (e.g. `5_10_crypto`) when `per_class_ok`, and pooled models keyed `f"{stop_pct}_{target_pct}"`
   otherwise. `for key, model in self.models.items()` has no filter matching a model's trained
   class to the current signal's `asset_class` feature, so e.g. an equity signal can be scored —
   and win the argmax — using a `5_10_crypto` classifier that never saw an equity row: an
   out-of-distribution prediction driving the final bracket choice. Fix: filter candidate models
   to those matching the current signal's asset class (falling back to the pooled/no-suffix key
   when no per-class model exists for that class) — read `train_tpsl_model.py`'s exact key-naming
   convention first and match it precisely, don't guess the format.

5. **(~line 1318) Broad `except Exception: continue` hides real prediction failures.**
   Wraps `model.predict_proba` calls — if `feature_metadata.json` and a `.pkl` model drift out of
   sync (e.g. a partially regenerated `data/tpsl_models/`), every candidate silently fails this
   try/except and the strategy falls back to the base signal with no error, indistinguishable
   from "no ML edge found." Fix: narrow the except to specific, genuinely-expected exceptions
   (e.g. a feature-vector shape mismatch), and log or re-raise anything else rather than
   silently continuing.

6. **(~line 1395) Unseen categorical value silently zeroed instead of raising.**
   `train_tpsl_model.py`'s own `_feature_columns()` docstring already calls this exact mismatch
   "a silent-failure risk (wrong columns score without raising)." An asset class or interval
   never seen in training leaves that one-hot segment all-zero, producing a real
   out-of-distribution feature vector fed straight into `predict_proba` with no signal anything
   is wrong. Fix: raise or log a clear warning when a categorical value isn't found in the saved
   training vocab, instead of silently proceeding with an all-zero segment.

7. **(~line 1358) Stale `confidence`/`expected_value` after bracket override.**
   `MLBracketStrategy` overrides `.stop`/`.target` with the ML-selected candidate but copies
   `confidence`/`expected_value` verbatim from `base_sig` — now stale relative to the new
   risk/reward. `kairos_signals.py`'s EV-based sort/gate (`_ev_pct_value`, `min_ev_pct`
   filtering) reads `sig.expected_value` directly, so a swapped-in wider/narrower bracket gets
   selected/ranked using an EV number describing a trade that no longer exists. Fix: recompute
   `confidence`/`expected_value` for the new `Signal` from the ML-chosen candidate's own
   `predicted_p_win`/`predicted_ev` (already computed during candidate selection) instead of
   copying `base_sig`'s stale values.

8. **(`tests/unit/test_ml_strategies.py` ~line 725) Zero real test coverage of the core
   algorithm.** `TestMLBracketStrategy` has exactly two tests — constructor-raises-on-missing-
   models, and base-strategy-returns-None passthrough — neither ever supplies a real
   `context`, so neither reaches feature vectorization or the argmax-EV loop. A regression there
   (reordered feature columns, wrong EV sign, broken metadata dict) would ship with green tests,
   exactly like items 1-7 above did. Fix: add tests that actually exercise the fixed core
   algorithm — supply realistic `context`/features/mocked-or-fixture models, and add a
   regression test per fix above (FLAT passthrough, asset-class filtering, recomputed EV/
   confidence, narrowed exception handling, unseen-category warning).

**Acceptance criteria:**
- [ ] All 7 correctness bugs above (items 1-7) are fixed, each verifiable independently.
- [ ] Item 8: new tests exercise the real EV-selection algorithm end-to-end (not just the two
  existing pass-through tests), plus one regression test per fix.
- [ ] Existing 2 tests in `TestMLBracketStrategy` still pass unchanged.
- [ ] No fix for one item silently breaks another — e.g. verify the FLAT-direction fix (3) and
  the corrected context-key fix (1) compose correctly (a FLAT signal from a correctly-identified
  ticker still passes through unchanged).

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_ml.py` and `tests/unit/test_ml_strategies.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Commit with "E21-S06" in the subject line. No Co-Authored-By trailer.
- [ ] Do NOT edit `docs/todo.md`.
