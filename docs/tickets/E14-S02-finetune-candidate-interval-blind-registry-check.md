# E14-S02 — Fix `select_finetune_candidate`'s interval-blind `already_registered` check

**Goal:** E14-S01's live verification found that `--stage finetune_next` with auto-selection (no `--assets`) finds ZERO candidates for any asset combination that already has a `finetuned_models` row at a DIFFERENT interval — defeating the table's own `UNIQUE(assets, interval)` design, which explicitly allows independent finetuning per interval. Fix the query to filter by interval.

**Context:**
- Read `docs/playbooks/hourly-finetuning.md`'s bug writeup (written by E14-S01, live-verified 2026-08-20) for the full root-cause narrative and the exact repro (`ZW=F` had a pre-existing `1d|failed` `finetuned_models` row, which silently made it invisible to `1h` auto-selection even though `ZW=F` had fresh `1h` oracle+base results ready).
- `strategy/kairos_pipeline.py`, `select_finetune_candidate()` (~line 1240-1400, search `def select_finetune_candidate`). Two identical bugs, both in this function:
  1. Line ~1322-1324 (standard candidate ranking):
     ```python
     already_registered = {
         r[0] for r in conn.execute("SELECT assets FROM finetuned_models").fetchall()
     }
     ```
     This pools `assets` (sorted-assets-CSV) across every row in the table regardless of `interval`. Then line ~1329: `if assets_sorted in already_registered: continue` — an asset combo registered at ANY interval, in ANY status (training/accepted/rejected/failed), permanently excludes it from candidacy at every OTHER interval too.
  2. Line ~1357 (the `priority_assets` branch): same `if assets_sorted in already_registered: continue` pattern, same bug, same fix needed.
- The table itself already has the right key: `UNIQUE(assets, interval)` (schema, ~line 208-223) — the bug is purely in this Python-side set construction not respecting that same key.
- `select_finetune_candidate()` is called from `run_stage_finetune_next()`, which is itself always called with a specific `interval` in scope — confirm the function signature already receives `interval` as a parameter (it does — check the `profiles.items()` loop already iterates `(assets, interval, backtest_period)` tuples per-profile, and the caller in `run_stage_finetune_next` presumably passes/filters by a target interval too; read enough of the surrounding code to confirm exactly how `interval` flows through before editing, since the fix must key `already_registered` by `(assets, interval)`, matching each candidate's own `interval` — not a single global interval, since `profiles` itself can contain multiple intervals in one call if `run_stage_finetune_next` doesn't pre-filter).

**Acceptance criteria:**
- [ ] `already_registered` becomes a set of `(assets, interval)` tuples: `{(r[0], r[1]) for r in conn.execute("SELECT assets, interval FROM finetuned_models").fetchall()}`.
- [ ] Both check sites become `if (assets_sorted, interval) in already_registered: continue` (using each candidate's own `interval` from its `(assets, interval, backtest_period)` profile key, not a single outer-scope interval).
- [ ] Unit test in `tests/unit/test_pipeline_auto.py` (check for an existing `TestSelectFinetuneCandidate`-style class to extend, otherwise add one): seed `finetuned_models` with a `(assets='ZW=F', interval='1d', status='failed')` row and matching `1h` `oracle_results`/`model_results` rows for `ZW=F`; call `select_finetune_candidate(conn, interval='1h', ...)` (or however the function is actually invoked — check its real signature) and assert `ZW=F` IS returned as a candidate at `1h` despite the `1d` registry row existing. Also assert the INVERSE still holds: a `(ZW=F, 1h)` registry row (any status) DOES exclude `ZW=F` from `1h` candidacy (regression guard for the fix not being too permissive).
- [ ] Existing tests covering `select_finetune_candidate`'s current (single-interval) behavior still pass unmodified — this fix only changes cross-interval blindness, not same-interval exclusion.
- [ ] Update `docs/playbooks/hourly-finetuning.md`'s bug section: mark it fixed (date + this ticket ID) rather than leaving "not fixed" language in place, and note that auto-selection (no `--assets`) can now be used going forward instead of the manual `--assets` re-queue workaround.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py`).
- [ ] New + existing tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] `docs/playbooks/hourly-finetuning.md` updated.
- [ ] Changes committed and `docs/todo.md` E14-S02 item checked off.
