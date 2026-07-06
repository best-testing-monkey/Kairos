# E7-S04: Implementation Shortfall

## Goal
Implement adaptive execution strategy choosing between immediate-fill and TWAP based on predicted drift vs impact.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.3
- Files to modify/create: strategy/kairos_execution.py, tests/unit/test_execution_algos.py
- Relevant existing code: TWAPExecutionStrategy from E7-S03
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- ImplementationShortfallStrategy wrapper with impact_bps parameter (default 5)
- Chooses between immediate-fill and TWAP per signal based on decision rule
- Decision: Kronos-predicted drift over execution horizon vs assumed impact cost (impact_bps param)
- Fast/immediate when drift adverse (drift < -impact_bps), patient/TWAP when favorable (drift > impact_bps), verified in `test_impl_shortfall_immediate_adverse_drift`
- Decision boundary tested at drift = impact_bps (indifference point), verified in `test_impl_shortfall_indifference_point`
- Both branches (immediate and TWAP) exercised in tests, verified in `test_impl_shortfall_both_branches`

## Definition of done
- Unit tests in `tests/unit/test_execution_algos.py` pass via `uv run --with pytest python -m pytest tests/unit/test_execution_algos.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
