# E9-S27 — End-to-end integration smoke test

**Goal:** Prove the whole pipeline (precompute → closure → replay) hangs together over a small synthetic window, with only true external boundaries mocked.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §5's "Replay vs. live sanity check" row.
- Read `strategy/kairos_signal_replay.py` (own module — ALL prior outputs: schema, unpack, interval resolution, max-adverse-excursion, closure computation, replay loop, CLI). This is the final integration story for this epic.
- Read `tests/unit/test_kairos_papertrade_leverage_regression.py`'s `_run_main`/`_FakeBarsProvider`/mocking patterns for this codebase's established house style on building a small synthetic fixture without live GPU/network — mirror that discipline (mock exactly the external boundary, let everything else run for real).

**Acceptance criteria:**
- [ ] One integration test in `tests/unit/test_kairos_signal_replay.py` that:
  1. Builds a small synthetic `signals_cache` table (a handful of signals across 2-3 `as_of` timestamps, at least 2 distinct tickers, at least one FLAT-direction row to prove exclusion still works end-to-end) in a throwaway sqlite3 connection.
  2. Calls `unpack_signals_cache_to_papertrade_signals` → `compute_closures_for_window` (with `price_cache.get_price_data` MOCKED to return synthetic bars — this is the only external boundary mocked, matching this session's `_FakeBarsProvider` precedent; do NOT mock any of this module's own functions) → `replay(...)`.
  3. Asserts: no exception raised anywhere in the chain; the returned metrics dict has all the keys required by E8-S23's DoD (`total_profit_eur`, `pct_profit`, `num_trades`, `pct_max_drawdown`); `num_trades` is consistent with how many of the synthetic signals were actually `SELECTED` by `allocate()` under the test's chosen `AllocationConfig` (verify this by also inspecting `papertrade_signals`/`papertrade_signals_closure` row counts, not just trusting the final number).
- [ ] This test intentionally exercises the FULL module end-to-end — the point is to catch integration seams (e.g. a function signature mismatch between two stories implemented independently) that per-story unit tests with narrower mocks could miss.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Integration smoke test passes.
- [ ] Changes committed and `docs/todo.md` E9-S27 item checked off — this is the LAST story in this epic; after this commit, every item under "Epic 6" through "Epic 9" in `docs/todo.md` must be `[x]`.
