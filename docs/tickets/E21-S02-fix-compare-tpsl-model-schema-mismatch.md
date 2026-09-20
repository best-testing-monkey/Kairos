# E21-S02 — Fix `compare_tpsl_model.py`'s nonexistent `c.as_of` column

**Goal:** `scripts/compare_tpsl_model.py`'s `_get_validation_window()` (~line 71) references a
column, `c.as_of`, that does not exist in the real `tpsl_label_candidates` table — a guaranteed
crash the moment this script runs against real data. Found by code review; masked in tests
because the test file hand-builds its own fixture table with an `as_of` column that doesn't
match the real schema.

**Context:**
- Real schema, created by `_ensure_tpsl_label_candidates_table()` in `scripts/tpsl_label_grid.py`
  (~line 52-63): columns are `signal_id`, `stop_pct`, `target_pct`, `resolved`,
  `hit_target_first`, `interval_used`, `engine_version`, `computed_at` — **no `as_of` column**.
- `as_of` lives on `papertrade_signals` (the table `tpsl_label_candidates.signal_id` references),
  per its schema in `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.1.
- Fix: `_get_validation_window()` needs to `JOIN tpsl_label_candidates c ON ... JOIN
  papertrade_signals s ON c.signal_id = s.signal_id` and read `s.as_of`, not `c.as_of`. Read the
  current SQL in full before editing — verify the exact join/column names against both tables'
  real, current schemas rather than assuming this description is still 100% accurate.
- **Test gap, fix it too**: `tests/unit/test_compare_tpsl_model.py` currently hand-builds a
  fixture table with its own `as_of` column, which is why this bug shipped with green tests.
  Rewrite (or add to) that fixture so it's built from the REAL `CREATE TABLE` statements in
  `tpsl_label_grid.py`/`papertrade_signals`' actual schema (import and call the real
  table-creation function rather than hand-rolling a diverging one), so a future schema drift
  like this one is caught automatically.

**Acceptance criteria:**
- [ ] `_get_validation_window()` (or wherever the query moved to) correctly joins to
  `papertrade_signals` for `as_of` — no reference to a nonexistent `c.as_of` column anywhere in
  `scripts/compare_tpsl_model.py`.
- [ ] Test fixture rebuilt from the real table-creation function(s), not a hand-rolled schema —
  verify by confirming the test would have failed against the buggy code before your fix.
- [ ] Unit test: `_get_validation_window()` (or equivalent) runs successfully against a
  realistically-shaped fixture DB and returns a sane date range.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `scripts/compare_tpsl_model.py`.
- [ ] New/updated tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Commit with "E21-S02" in the subject line. No Co-Authored-By trailer.
- [ ] Do NOT edit `docs/todo.md`.
