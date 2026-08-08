# E5-S14 — Frozen-fixture MTM repro test

**Goal:** Add `tests/unit/test_kairos_papertrade_mtm_repro.py` that replays the frozen 2026-07-26 fixture through the MTM path and asserts convergence invariants.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §6.2.
- Read `tests/unit/test_kairos_papertrade_loss_repro.py` for fixture loading pattern (`_phantom_client`, `FIXTURE_DB_V2_FIXED`, `ACCOUNT_NAME_V2_FIXED`, `CAPITAL`).
- Read `strategy/kairos_papertrade.py` `compute_final_metrics()` (output of E4-S12) and the `kairos_mtm_daily` table schema.
- Create/modify `tests/unit/test_kairos_papertrade_mtm_repro.py`.

**Acceptance criteria:**
- [ ] Test file follows the same fixture-copy pattern as `test_kairos_papertrade_loss_repro.py` to avoid mutating the checked-in DB.
- [ ] It runs `compute_final_metrics` on the frozen fixture and then inspects `kairos_mtm_daily`.
- [ ] Asserts `kairos_mtm_daily` row count equals the number of trading days in the replay window.
- [ ] Asserts final MTM equity equals final corrected closed-trade equity (all positions removed at window end, so curves converge at the endpoint).
- [ ] Asserts `mtm_max_drawdown_pct >= pct_max_drawdown`.
- [ ] Test is skipped with `pytest.skip` if the fixture DB is missing.
- [ ] Test passes with `uv run --with pytest python -m pytest tests/unit/test_kairos_papertrade_mtm_repro.py -q` when the fixture is present.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Repro test passes (or skips gracefully if fixture absent).
- [ ] Changes committed and `docs/todo.md` E5-S14 item checked off.
