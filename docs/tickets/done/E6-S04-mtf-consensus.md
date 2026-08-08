# E6-S04: MTF Consensus

## Goal
Implement multi-timeframe consensus filter requiring majority agreement across {1d, 3d, 1w}.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.1
- Files to modify/create: strategy/kairos_backtest.py, tests/unit/test_technical_filters.py
- Relevant existing code: kairos.calendar for boundary-correct resampling
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- MTFConsensusStrategy filter wrapper
- Resamples history to {1d, 3d, 1w} via kairos.calendar (boundary-correct, not naive date-based)
- Computes trend sign per frame (EMA20 vs EMA50 equivalent)
- Requires >=2/3 agreement with signal direction for pass-through
- Vetoes signals with <2/3 agreement, verified in `test_mtf_2_agree_passes`
- Blocks with 1 agreement, verified in `test_mtf_1_agree_vetoes`
- Resampling is boundary-correct via kairos.calendar, verified in `test_mtf_resampling_boundaries`
- 2/3 vote logic tested (2-agree passes, 1-agree vetoes)

## Definition of done
- Unit tests in `tests/unit/test_technical_filters.py` pass via `uv run --with pytest python -m pytest tests/unit/test_technical_filters.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
