# E3-S07 — Extend AllocationConfig with leverage fields

**Goal:** Add `max_leverage` and `margin_utilization_cap` fields to `AllocationConfig` in `strategy/allocation.py` with defaults that preserve current behavior.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.3.
- Read `strategy/allocation.py` `AllocationConfig` dataclass (around line 45).
- Modify `strategy/allocation.py` only.

**Acceptance criteria:**
- [ ] `AllocationConfig` gains `max_leverage: float = 1.0`.
- [ ] `AllocationConfig` gains `margin_utilization_cap: float = 0.8`.
- [ ] Existing default `gross_cap_pct: float = 100` is unchanged.
- [ ] Existing unit tests in `tests/unit/` that construct `AllocationConfig` still pass without modification.
- [ ] New `AllocationConfig()` instance has `.max_leverage == 1.0` and `.margin_utilization_cap == 0.8`.

**Definition of done:**
- [ ] `flake8` and `mypy` pass on `strategy/allocation.py`.
- [ ] `uv run --with pytest python -m pytest tests/unit/test_kairos_allocation.py tests/unit/test_allocation.py -q` passes (run whichever allocation test files exist; if none exist, run the full unit suite).
- [ ] Changes committed and `docs/todo.md` E3-S07 item checked off.
