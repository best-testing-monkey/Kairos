# E4-S11 — Implement liquidation execution path

**Goal:** Implement daily liquidation in `strategy/kairos_papertrade.py`: trigger check, direct-DB close, corrected cash, Telegram notification, and ruined flag.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.4.
- Read `strategy/kairos_papertrade.py` `main()` day loop and `remove_all_open_positions()` for the direct-DB pattern.
- Read `strategy/kairos_mtm.py` `liquidation_check` (output of E2-S05).
- Read `strategy/kairos_margin.py` (output of E1-S02).
- Modify `strategy/kairos_papertrade.py`.

**Acceptance criteria:**
- [ ] After the MTM snapshot each day, `liquidation_check` is called; if it returns tickers, those positions are liquidated.
- [ ] Liquidation writes phantom rows with `status='liquidated'` (enum already allows it) and nulls `orders.position_id` first to satisfy the RESTRICT FK.
- [ ] The corrected cash effect is applied Kairos-side (direction-aware gross PnL minus exit costs minus fx); phantom's `PositionAPI.close()` is never called for liquidations.
- [ ] One `_notify()` Telegram line is sent per liquidation event.
- [ ] `negative_balance_protection` clamps equity at `0.0`; a run-level `ruined=True` flag stops opening new positions for the rest of the window (existing positions still resolve via SL/TP).
- [ ] Liquidated positions are excluded from normal close processing and counted in `mtm_liquidation_events`.
- [ ] Unit/integration tests verify the trigger, status, cash effect, and ruined flag using a mocked or small fixture DB.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Liquidation tests pass.
- [ ] Changes committed and `docs/todo.md` E4-S11 item checked off.
