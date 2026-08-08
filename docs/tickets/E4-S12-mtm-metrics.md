# E4-S12 — Add MTM metrics block and extend reconciliation

**Goal:** Add MTM-derived summary metrics to `compute_final_metrics()` and extend `_reconcile_cash_and_log()` with MTM-aware checks.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.6.
- Read `strategy/kairos_papertrade.py` `compute_final_metrics()` (around line 1151), `_reconcile_cash_and_log()` (around line 1114), `build_closed_trade_equity_curve()`.
- Read `strategy/kairos_mtm.py` `DailySnapshot` (output of E2-S03).
- Modify `strategy/kairos_papertrade.py`.

**Acceptance criteria:**
- [ ] `compute_final_metrics()` reads `kairos_mtm_daily` rows for the account and computes:
  - `mtm_total_return_pct`
  - `mtm_max_drawdown_pct`
  - `mtm_sharpe`
  - `mtm_margin_utilization_peak`
  - `mtm_financing_total_eur`
  - `mtm_liquidation_events`
  - `mtm_ruined`
- [ ] Existing closed-trade metrics (`total_profit_eur`, `pct_profit`, etc.) remain unchanged.
- [ ] Sharpe is computed from daily equity differences (mean / std * sqrt(252)) or by reusing phantom's `calculate_metrics` on an equity-point curve built from `kairos_mtm_daily`.
- [ ] `_reconcile_cash_and_log()` is extended to log `corrected_cash` plus open `unrealized_pnl` vs phantom raw equity, still warning-only.
- [ ] Unit tests verify the new metric keys exist and have plausible values on a synthetic `kairos_mtm_daily` table.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] New metrics tests pass.
- [ ] `uv run --with pytest python -m pytest tests/unit/test_kairos_papertrade_loss_repro.py -q` still passes.
- [ ] Changes committed and `docs/todo.md` E4-S12 item checked off.
