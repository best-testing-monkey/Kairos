# E11-S02: SCHEMA_ERROR validation

## Goal
Add `validate_candidate(c: Candidate) -> Optional[str]` to `strategy/allocation.py`, returning `"SCHEMA_ERROR"` (or `None` if valid) per RFC §3's rejection rule.

## Context
- Read: docs/rfc_allocation_sheet.md §3 (last paragraph before "Migration note"), §4.4 (gate step — `reject(c, SCHEMA_ERROR)`)
- Files to modify: strategy/allocation.py (add to the module from E11-S01); tests/unit/test_allocation.py (append)
- Depends on: E11-S01 (`Candidate` dataclass)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `validate_candidate` returns `"SCHEMA_ERROR"` if: any required field (strategy, ticker, direction, entry, stop, target, ev_pct, base_win_rate, n, backtest_period, sharpe) is `None` or non-finite (use `math.isfinite`, reject NaN/inf)
- `validate_candidate` returns `"SCHEMA_ERROR"` if direction/stop/target placement is inconsistent: `direction == "long"` requires `stop < entry < target`; `direction == "short"` requires `target < entry < stop`
- `validate_candidate` returns `None` for a well-formed long and a well-formed short candidate (both cases tested)
- Tests cover: missing field, NaN field, long with stop/target swapped, short with stop/target swapped, and the two valid cases — one test each, named after the condition (e.g. `test_long_with_swapped_stop_target_is_schema_error`)

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
