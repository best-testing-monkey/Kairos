# E10-S03: run_stage_auto chaining with resumability and failure isolation

## Goal
Implement `run_stage_auto()` chaining universe → correlation → per-group oracle → per-group base for each interval, with skip-if-done resumability and per-group failure isolation.

## Context
- Read: strategy/DESIGN_DOC_pipeline_automation.md §2.2 (the exact algorithm, skip-log format, failure summary, exit-code rule, runs-table bookkeeping)
- Files to modify: strategy/kairos_pipeline.py (add `run_stage_auto(conn, intervals, backtest_period, asset_class_filter, pred_samples, min_sharpe, min_signals, force, skip_universe) -> pd.DataFrame`); tests/unit/test_pipeline_auto.py (append)
- Reuse UNCHANGED: `run_stage_universe`, `run_stage_correlation`, `run_stage_oracle`, `run_stage_model`, `build_viability_report` (E10-S02). Read `suggested_groups` reading code in the existing oracle stage (`--group_id` path) to copy the group-fetch query.
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Tests monkeypatch `run_backtest_subprocess` (canned payload) and the universe/correlation stage functions where needed; NO GPU/model/network
- Chaining order per interval: universe → correlation → for each suggested group (oracle then base) — call order asserted via a recording stub (`test_auto_chaining_order`)
- `--intervals ["1d","1h"]` → the whole chain runs once per interval (`test_auto_multi_interval`)
- Resumability: pre-inserted `oracle_results` rows matching (assets_key, interval, backtest_period) → oracle call skipped for that group, executed with force=True (`test_auto_resume_skip`, `test_auto_force_reruns`)
- Failure isolation: stub raising RuntimeError for one group → remaining groups still run; failure summary contains exactly that group; function does not raise (`test_auto_failure_isolation`)
- One `runs` row inserted with stage="auto" and params_json containing intervals/backtest_period (`test_auto_runs_bookkeeping`)
- skip_universe=True with existing universe/correlation runs → universe/correlation functions NOT called (`test_auto_skip_universe`)
- Ends by calling build_viability_report and returning its DataFrame

## Definition of done
- `timeout 120 uv run --with pytest python -m pytest tests/unit/test_pipeline_auto.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
