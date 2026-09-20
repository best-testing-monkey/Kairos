# E21-S04 — Fix `kairos_tpsl_features.py`: timezone crash + division-by-zero

**Goal:** Two hardening bugs in `strategy/kairos_tpsl_features.py`, found by code review, both
missed because existing tests only use synthetic tz-naive/well-formed fixtures.

**Context:**

**Bug 1 (crash, ~line 50):** `pd.Timestamp(as_of)` builds a tz-naive `Timestamp` compared
against `history.index`, which is tz-aware when history is fetched via
`price_cache.get_price_data` directly (as `scripts/compare_tpsl_model.py` and
`scripts/tpsl_label_grid.py` do, bypassing `kairos.data.fetch_data_raw`'s tz-stripping
normalization step). Real callers hit
`history[history.index <= as_of_date]` → `TypeError: Invalid comparison between
dtype=datetime64[ns, tz] and Timestamp`.
`tests/unit/test_tpsl_features.py` only uses tz-naive `pd.date_range` fixtures, so this path is
never exercised. Fix: normalize `as_of`'s tz-awareness to match `history.index` before comparing
— if `history.index` is tz-aware, localize/convert `as_of` to match; if naive, leave it naive.
**Do not blindly strip tz info** — this repo has a documented history of exactly that kind of bug
mislabeling bars by hours (see CLAUDE.md's "price_cache: DST-ambiguous-time crash + crypto tz
mislabeling" section) — match, don't discard.

**Bug 2 (silent NaN/inf, ~line 87):** `range_position` divides by `(recent_high - recent_low)`
with no zero-guard. Any 20-bar window where high==low on every bar (a halted or tightly-pegged
instrument) produces NaN (0/0) or inf, which flows unguarded through feature vectorization
straight into `model.predict_proba()` as a raw feature value, corrupting that prediction rather
than raising or handling it explicitly. No test exercises a flat-range case. Fix: guard the
denominator — when `recent_high == recent_low` (or the difference is below a small epsilon),
return a defined, documented sentinel value (e.g. `0.5`, representing "middle of a degenerate
range") instead of computing the division.

**Acceptance criteria:**
- [ ] `extract_features()` (or wherever the `as_of`/history comparison lives) works correctly for
  both tz-naive and tz-aware `history` inputs — no `TypeError`.
- [ ] Unit test: tz-aware `history` fixture (e.g. built with `tz_localize("America/New_York")` or
  similar) passed to `extract_features()` — no crash, correct truncation behavior preserved (the
  existing no-lookahead invariant test from E18-S02 must still pass under tz-aware input too).
- [ ] `range_position` (or its containing feature function) returns a defined value, not NaN/inf,
  when `recent_high == recent_low`.
- [ ] Unit test: a fixture with a flat 20-bar range (high==low on every bar) produces the
  documented sentinel value, not NaN/inf — assert with `math.isnan`/`math.isinf` checks that it's
  NOT either.
- [ ] All existing `test_tpsl_features.py` tests still pass unchanged.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_tpsl_features.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Commit with "E21-S04" in the subject line. No Co-Authored-By trailer.
- [ ] Do NOT edit `docs/todo.md`.
