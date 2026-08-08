# E6-S03: OBV Confirmation

## Goal
Implement On-Balance Volume confirmation filter that validates signal direction against volume trend.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.1
- Files to modify/create: strategy/kairos_backtest.py, tests/unit/test_technical_filters.py
- Relevant existing code: VolumeConfirmationStrategy pattern (uses predicted volume) in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- OBVConfirmationStrategy filter wrapper
- Computes OBV(20-slope) from realized volume
- Vetoes signals when OBV slope sign disagrees with signal direction, verified in `test_obv_slope_disagreement_veto`
- Passes signals when OBV slope agrees with direction, verified in `test_obv_slope_agreement_pass`
- Flat OBV (near-zero slope) passes through without veto, verified in `test_obv_flat_slope_passthrough`
- Complements VolumeConfirmationStrategy (which uses predicted volume), verified via different data source (realized vs predicted)

## Definition of done
- Unit tests in `tests/unit/test_technical_filters.py` pass via `uv run --with pytest python -m pytest tests/unit/test_technical_filters.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
