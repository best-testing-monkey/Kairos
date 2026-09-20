# E21-S03 — Fix `tpsl_label_grid.py`'s partial-coverage resume bug

**Goal:** The resume query in `scripts/tpsl_label_grid.py` (~line 272) treats a signal as fully
covered the moment ANY of its 9 `(stop_pct, target_pct)` candidate rows exists, not all 9. Found
by code review — an interrupted run permanently and silently skips affected signals on every
future re-run.

**Context:**
- Current logic: `LEFT JOIN tpsl_label_candidates c ... WHERE c.signal_id IS NULL` — this
  excludes a signal from `uncovered_signals` as soon as **1 of 9** grid-cell rows exists for it.
  If the script is killed (OOM, Ctrl-C, reboot, rate limit) after writing 2 of 9 rows for a
  signal, every future re-run silently skips that signal forever, leaving
  `scripts/train_tpsl_model.py` to train on a systematically incomplete subset with no warning.
- `tests/unit/test_tpsl_label_grid.py`'s `test_idempotent_re_run` only exercises the per-signal
  `resolve_candidate()`-equivalent function directly, never this `main()`-level resume query —
  that's why this shipped green. Fix the resume query AND add a test that actually exercises it.
- Fix shape: the resume check needs to count matching rows per signal for the current
  `engine_version` and only treat a signal as covered when the count reaches the full grid size
  (9, or however many candidates `STOP_PCT_GRID × TARGET_PCT_GRID` currently produces — derive
  the expected count from those constants, don't hardcode `9` if the grid size could change).
  Something like a `GROUP BY signal_id HAVING COUNT(*) < <expected_count>` join, or a per-signal
  existence check against the full candidate set. Read the current query in full before deciding
  the exact SQL shape.

**Acceptance criteria:**
- [ ] Resume logic only treats a signal as fully covered when it has a row for every grid
  candidate under the current `engine_version` — a signal with 1-8 of 9 rows is still returned
  for (re-)processing.
- [ ] Unit test: seed a fixture DB with a signal that has exactly 2 of 9 candidate rows already
  written (simulating an interrupted run), run the resume-aware main loop, and assert the
  remaining 7 candidates get computed and written — not skipped.
- [ ] Unit test: a signal with all 9 rows already present is correctly skipped (no wasted
  recomputation) — confirms the fix didn't just make everything always reprocess.
- [ ] Existing `test_idempotent_re_run` still passes unchanged.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `scripts/tpsl_label_grid.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Commit with "E21-S03" in the subject line. No Co-Authored-By trailer.
- [ ] Do NOT edit `docs/todo.md`.
