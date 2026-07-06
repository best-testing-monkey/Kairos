# E7-S02: CVD Divergence

## Goal
Implement cumulative volume delta divergence strategy trading price-volume disagreements.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.2
- Files to modify/create: strategy/kairos_execution.py, tests/unit/test_execution_algos.py
- Relevant existing code: None yet (CVD computation)
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- CVDDivergenceStrategy standalone strategy
- Computes cumulative volume delta from daily bars (volume signed by close-vs-open)
- Trades divergence between 20-day CVD slope and price slope
- Gated on Kronos agreement
- Sign convention tested: up close = positive volume delta, verified in `test_cvd_sign_convention`
- Divergence detection on planted fixture (CVD falling while price rising, or vice versa), verified in `test_cvd_divergence_detection`
- Correctly identifies both up-divergence and down-divergence scenarios

## Definition of done
- Unit tests in `tests/unit/test_execution_algos.py` pass via `uv run --with pytest python -m pytest tests/unit/test_execution_algos.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
