# E10-S01: Period-to-weeks helper

## Goal
Add a `_period_to_weeks(period: str) -> float` helper reusing the existing backtest-period parser, as the basis for the signals-per-week column.

## Context
- Read: strategy/DESIGN_DOC_pipeline_automation.md §3.1 (signals_per_week bullet) and §7
- Files to modify: strategy/kairos_strategies.py (add next to `_period_to_bars` — read that function first and reuse its period-string parsing; do NOT write a second regex/parser)
- Files to create: tests/unit/test_pipeline_auto.py (NEW — this story creates the file; later stories append)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `_period_to_weeks("6m")` ≈ 26.09, `"1m"` ≈ 4.35 (365.25/12 days per month, 7 days/week), `"2w"` == 2.0, `"1y"` ≈ 52.18 — each asserted to 1e-2 in `test_period_to_weeks_values`
- Invalid period string raises the same error type `_period_to_bars` raises for invalid input, verified in `test_period_to_weeks_invalid`
- `grep -n "def _period_to_weeks" strategy/kairos_strategies.py` shows a line

## Definition of done
- `timeout 120 uv run --with pytest python -m pytest tests/unit/test_pipeline_auto.py -q` passes
- No regression: `timeout 120 uv run --with pytest python -m pytest tests/unit/test_backtest_engine.py -q`
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
