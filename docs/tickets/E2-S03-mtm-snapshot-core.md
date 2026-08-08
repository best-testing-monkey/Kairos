# E2-S03 — Build daily MTM snapshot dataclasses and pure math

**Goal:** Create `strategy/kairos_mtm.py` with `OpenPositionView`, `DailySnapshot`, `unrealized_pnl`, and `compute_daily_snapshot` — all pure functions over plain data.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.2.
- Read `strategy/kairos_margin.py` (output of E1-S02) for `MarginConfig` / `MarginClass` / `classify_symbol`.
- Create/modify `strategy/kairos_mtm.py`.
- Create/modify `tests/unit/test_kairos_mtm.py` (snapshot section).

**Acceptance criteria:**
- [ ] `OpenPositionView` dataclass has fields exactly: `ticker`, `direction`, `entry_price`, `quantity`, `entry_costs`.
- [ ] `DailySnapshot` dataclass has fields exactly: `date`, `cash`, `unrealized_pnl`, `equity`, `gross_notional`, `initial_margin_used`, `maintenance_margin_used`, `free_margin`, `margin_utilization`, `financing_accrued_day`, `liquidations`.
- [ ] `unrealized_pnl(pos, close_price)` is direction-aware:
  - long: `(close - entry) * qty`
  - short: `(entry - close) * qty`
- [ ] `compute_daily_snapshot(positions, bars_by_ticker, cash, cfg) -> DailySnapshot`:
  - Uses `classify_symbol` per position to pick its `MarginClass`.
  - `gross_notional = sum(entry_price * quantity)` for all open positions.
  - `initial_margin_used = sum(notional * class.initial_margin_pct / 100)`.
  - `maintenance_margin_used = sum(notional * class.maintenance_margin_pct / 100)`.
  - `unrealized_pnl = sum(unrealized_pnl(pos, close_price))`.
  - `equity = cash + unrealized_pnl`.
  - `free_margin = equity - initial_margin_used`.
  - `margin_utilization = initial_margin_used / equity` when `equity > 0`, else `0.0`.
  - `financing_accrued_day` and `liquidations` start at `0.0` / `0` in this story.
- [ ] Unit test with a 2-position fixture asserts exact values for all `DailySnapshot` fields.
- [ ] Module imports no `phantom`, GPU, or network libraries.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] New snapshot unit tests pass.
- [ ] Changes committed and `docs/todo.md` E2-S03 item checked off.
