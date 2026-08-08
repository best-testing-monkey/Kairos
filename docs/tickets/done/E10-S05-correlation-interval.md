# E10-S05: Interval-aware correlation stage

## Goal
Verify and, if needed, fix the correlation stage (stage 2) to honor a non-1d interval when fetching closes.

## Context
- Read: strategy/DESIGN_DOC_pipeline_automation.md §2.2 step 2 (the ⚠ implementation item); strategy/kairos_pipeline.py `run_stage_correlation` and its price-fetch path; `calendar_days_for_bars` and `fetch_data_raw` in strategy/kairos_strategies.py (reuse — do not invent new bar/calendar math)
- Files to modify: strategy/kairos_pipeline.py; tests/unit/test_pipeline_auto.py (append)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- FIRST determine current behavior: if `run_stage_correlation` already threads interval into its data fetch, write the verification test and report "no change needed" in the commit message; otherwise add an `interval="1d"` parameter and thread it through the fetch, using `calendar_days_for_bars` for the window so 1h bars cover an equivalent calendar window
- `run_stage_auto` passes its per-iteration interval into `run_stage_correlation` (update E10-S03's call site if the signature changed)
- Test with a monkeypatched fetch function recording the interval argument: correlation at interval="1h" requests 1h data (`test_correlation_interval_threading`); default 1d behavior byte-identical for existing single-stage use (recording stub asserts same args as before when interval omitted)
- No network in tests (fetch is stubbed)

## Definition of done
- `timeout 120 uv run --with pytest python -m pytest tests/unit/test_pipeline_auto.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
