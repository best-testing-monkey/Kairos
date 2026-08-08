# E4-S10 — Wire admission check and locked-margin order semantics

**Goal:** Gate each day's orders through `admission_check` and implement locked-margin cash semantics for marginable asset classes.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.3.
- Read `strategy/kairos_papertrade.py` `main()` order placement block (around lines 1545-1560), `map_instrument_type()`.
- Read `strategy/kairos_mtm.py` `admission_check` (output of E2-S04) and `classify_symbol` (output of E1-S02).
- Read `strategy/allocation.py` `AllocationConfig` (output of E3-S07).
- Modify `strategy/kairos_papertrade.py`.

**Acceptance criteria:**
- [ ] Before placing an order, the day loop calls `admission_check` using the order notional (`alloc_eur`), ticker, current `DailySnapshot`, margin config, and allocation config.
- [ ] Rejected orders are skipped, a `MARGIN_REJECTED` log line is emitted, and the rejection is counted in metrics (see E4-S12).
- [ ] For `initial_margin_pct < 100` classes, only entry costs (commission + spread + slippage + fx) are debited from `corrected_cash` at fill; the full notional is no longer removed.
- [ ] For `initial_margin_pct == 100` classes (crypto spot), the full notional is still debited from `corrected_cash` at fill, preserving current behavior.
- [ ] When `max_leverage == 1.0`, the admission check is a no-op and cash handling is byte-identical to the legacy path.
- [ ] Unit tests mock `client.orders.place` and assert: accepted orders when under cap, rejected orders when over cap, correct cash debit for CFD vs spot classes.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] New admission/order unit tests pass.
- [ ] `uv run --with pytest python -m pytest tests/unit/test_kairos_papertrade_loss_repro.py -q` still passes.
- [ ] Changes committed and `docs/todo.md` E4-S10 item checked off.
