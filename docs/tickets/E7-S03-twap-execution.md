# E7-S03: TWAP Execution

## Goal
Implement time-weighted average price (TWAP) execution wrapper splitting entry across predicted path.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.3
- Files to modify/create: strategy/kairos_execution.py, tests/unit/test_execution_algos.py
- Relevant existing code: PathExecutionStrategy plumbing in strategy/kairos_execution.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- TWAPExecutionStrategy wrapper (reuses PathExecutionStrategy plumbing)
- Splits entry across first k in-day steps of predicted path
- Records per-slice fills in signal.metadata["fills"]
- Average fill becomes effective entry, verified in `test_twap_average_fill_effective_entry`
- Bracket recomputed off effective entry (not original entry), verified in `test_twap_bracket_from_effective_entry`
- Per-slice fills recorded and retrievable from metadata, verified in `test_twap_fills_in_metadata`

## Definition of done
- Unit tests in `tests/unit/test_execution_algos.py` pass via `uv run --with pytest python -m pytest tests/unit/test_execution_algos.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
