# E4-S09 — Maintain corrected cash and persist daily MTM snapshots

**Goal:** In `strategy/kairos_papertrade.py` day loop, maintain a Kairos-side `corrected_cash`, compute and persist a `DailySnapshot` per day, and accrue financing.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.2 and §4.5.
- Read `strategy/kairos_papertrade.py` `main()` day loop (around line 1533), `compute_corrected_realized_pnl()`, `_reconcile_cash_and_log()`, `_IntradayFallbackProvider`.
- Read `strategy/kairos_mtm.py` (outputs of E2-S03 and E2-S06) for `OpenPositionView`, `DailySnapshot`, `compute_daily_snapshot`, `compute_daily_financing_total`.
- Read `strategy/kairos_margin.py` (output of E1-S02) for `classify_symbol`.
- Modify `strategy/kairos_papertrade.py`.

**Acceptance criteria:**
- [ ] `main()` declares a `corrected_cash = args.capital` before the day loop and updates it on every order fill and every position close using the same direction-aware correction philosophy as `compute_corrected_realized_pnl`.
- [ ] A new SQLite table `kairos_mtm_daily` is created in `data/phantom_ledger/phantom.db` with the schema from §4.2.
- [ ] After each `runner.backtest()` call, the day loop fetches open positions and that day's closing price per ticker (reusing the same bars already fetched; cache per-day to avoid double fetching), computes a `DailySnapshot`, and inserts one row into `kairos_mtm_daily`.
- [ ] `financing_accrued_day` is populated from `compute_daily_financing_total`; `financing_accrued_total` is a running cumulative sum.
- [ ] `_reconcile_cash_and_log()` is extended to also log `corrected_cash` + open `unrealized_pnl` vs phantom raw equity, still as a warning-only check.
- [ ] When `max_leverage == 1.0`, the corrected-cash path is byte-identical to the legacy cash path for long-only/spot trades (no margin lock, no financing for spot).
- [ ] Existing frozen-fixture loss-repro tests still pass.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] `uv run --with pytest python -m pytest tests/unit/test_kairos_papertrade_loss_repro.py -q` passes.
- [ ] A new unit test verifies the `kairos_mtm_daily` schema and at least one synthetic insert.
- [ ] Changes committed and `docs/todo.md` E4-S09 item checked off.
