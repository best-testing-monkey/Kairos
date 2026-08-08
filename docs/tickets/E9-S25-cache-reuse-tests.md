# E9-S25 — Dedicated cache-reuse & engine_version-bump regression tests

**Goal:** Explicit, clearly-named regression tests for `--precompute`'s idempotency and `engine_version`-bump-forces-recompute behavior, per the design doc's testing plan.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §5 (Testing plan), specifically the "Cache reuse" and "`engine_version` bump forces recompute" rows.
- Read `strategy/kairos_signal_replay.py` (own module — all prior outputs: `_ensure_signal_replay_tables`, `unpack_signals_cache_to_papertrade_signals`, `compute_closures_for_window`).
- These behaviors may already be PARTIALLY exercised incidentally by earlier stories' own tests (E6-S18, E7-S21) — this story's job is to make sure both have an explicit, dedicated, clearly-named test proving them, not just an incidental side effect of another test passing.

**Acceptance criteria:**
- [ ] `test_precompute_is_idempotent` (or similar clear name): calls `unpack_signals_cache_to_papertrade_signals` + `compute_closures_for_window` twice over the exact same synthetic window/data. Asserts the SECOND pass inserts/recomputes zero rows — check both the return-value counts AND that `created_at`/`computed_at` timestamps on existing rows are unchanged between the two passes (proves rows weren't silently rewritten with new timestamps even if content is identical).
- [ ] `test_engine_version_bump_forces_recompute` (or similar clear name): calls `compute_closures_for_window` with `engine_version="v1"`, then again with `engine_version="v2"` over the same window. Asserts existing closure rows ARE recomputed the second time (row count of recomputed rows > 0, and `computed_at`/`engine_version` columns reflect the new value).
- [ ] Both tests use synthetic, hand-built fixture data (no live network/GPU), consistent with this module's established test style from earlier stories.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Both dedicated tests pass.
- [ ] Changes committed and `docs/todo.md` E9-S25 item checked off.
