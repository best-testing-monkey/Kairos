# E4-S02: VAR Lead-Lag

## Goal
Implement standalone strategy detecting lead-lag relationships via VAR(1) that generate cross-asset signals.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §4.2
- Files to modify/create: strategy/kairos_econometric.py, tests/unit/test_econometric.py
- Relevant existing code: _lagged_ols helper from E4-S01
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- VARLeadLagStrategy standalone strategy fitting VAR(1) via OLS on 3-asset return panel
- Detects when asset j's lagged return significantly (|t|>2) predicts asset i's return
- Yesterday's j-move implies i-move that agrees with Kronos direction for i triggers signal, verified in `test_var_leadlag_detection`
- Uses DynamicBracketStrategy-style bracket (from strategy/kairos_backtest.py pattern)
- On synthetic data with planted x->y lag-1 dependence, detects only that edge, verified in `test_var_leadlag_specificity`
- No signal when coefficient insignificant (|t| <= 2), verified in `test_var_leadlag_insignificant_threshold`

## Definition of done
- Unit tests in `tests/unit/test_econometric.py` pass via `uv run --with pytest python -m pytest tests/unit/test_econometric.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
