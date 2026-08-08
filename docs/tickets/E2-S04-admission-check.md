# E2-S04 — Add margin admission check

**Goal:** Add `admission_check` to `strategy/kairos_mtm.py` so the day loop can reject new orders that would breach the margin-utilization cap.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.3.
- Read `strategy/kairos_mtm.py` (output of E2-S03) and `strategy/kairos_margin.py` (output of E1-S02).
- Read `strategy/allocation.py` `AllocationConfig` (output of E3-S07 if available, otherwise read current version to see `max_leverage` / `margin_utilization_cap` fields).
- Modify `strategy/kairos_mtm.py`.
- Add tests to `tests/unit/test_kairos_mtm.py`.

**Acceptance criteria:**
- [ ] `admission_check(order_notional: float, ticker: str, account: DailySnapshot, cfg: MarginConfig, alloc_config: AllocationConfig) -> bool` is defined.
- [ ] When `alloc_config.max_leverage <= 1.0`, the function returns `True` immediately (no-op, preserves legacy behavior).
- [ ] Otherwise it computes the post-trade state assuming the new order is filled:
  - `new_gross_notional = account.gross_notional + order_notional`
  - `new_initial_margin_used = account.initial_margin_used + order_notional * class.initial_margin_pct / 100`
  - `new_equity = account.equity` (entry costs are handled by the caller)
- [ ] Returns `True` only if `new_initial_margin_used <= new_equity * alloc_config.margin_utilization_cap` AND `new_equity > 0`.
- [ ] Returns `False` for any breach; the caller must log `MARGIN_REJECTED` and skip the order.
- [ ] Unit tests cover: accept below cap, reject above cap, reject when equity <= 0, no-op when max_leverage == 1.0.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Admission unit tests pass.
- [ ] Changes committed and `docs/todo.md` E2-S04 item checked off.
