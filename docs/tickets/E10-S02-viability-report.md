# E10-S02: Viability report builder + table

## Goal
Implement `build_viability_report()` joining latest oracle and base results per (strategy, assets, interval, backtest_period), with full per-side metrics, signals_per_week, and viable flag; persist to a new `viability_report` SQLite table and CSV.

## Context
- Read: strategy/DESIGN_DOC_pipeline_automation.md §3 (ALL of it — columns are exact and per-side prefixed)
- Files to modify: strategy/kairos_pipeline.py (add module-level `build_viability_report(conn, intervals, backtest_period, min_sharpe=0.0, min_signals=3, asset_class_filter=None) -> pd.DataFrame`; add `viability_report` CREATE TABLE IF NOT EXISTS alongside existing schema statements; reuse `dump_csv` for `results/auto_viability_report_<ts>.csv`); tests/unit/test_pipeline_auto.py (append)
- Reuse: `asset_class_for(assets)` and `_period_to_weeks` from kairos_strategies.py (E10-S01); existing `oracle_results` / `model_results` schemas (read the CREATE TABLE statements in kairos_pipeline.py)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Columns exactly and in order: strategy_name, assets, asset_class, interval, backtest_period, oracle_sharpe, oracle_signals, oracle_win_rate, oracle_avg_pnl_per_trade, oracle_run_id, base_sharpe, base_signals, base_win_rate, base_avg_pnl_per_trade, base_run_id, base_model_path, signals_per_week, viable — verified in `test_report_columns_exact`
- Per-side metric columns generated generically from the results-table schema (prefixing), not a hardcoded metric list (identifying + signals_per_week + viable stay fixed) — verified by asserting a synthetic extra column in a fixture table appears prefixed
- Latest-run-wins: fixture DB with two oracle runs for the same key → only max run_id row used (`test_report_latest_run_wins`)
- Outer join: strategy only in oracle → base_* NaN and viable False; only in base → oracle_* NaN and viable False (`test_report_outer_join_nan_viable_false`)
- viable = oracle_sharpe > min_sharpe AND base_sharpe > min_sharpe AND min(oracle_signals, base_signals) >= min_signals; boundary cases tested (`test_report_viability_gating`)
- signals_per_week = base_signals / _period_to_weeks(backtest_period), verified to 1e-6 on a fixture row
- Sort: viable first, then base_sharpe desc (`test_report_sort_order`)
- `viability_report` table row count == DataFrame length after a build; CSV written with exact header (`test_report_persistence`)
- All fixture DBs are temp SQLite files built in the test (no GPU/network)

## Definition of done
- `timeout 120 uv run --with pytest python -m pytest tests/unit/test_pipeline_auto.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
