# E4-S01: Lagged OLS and ARIMA Disagreement

## Goal
Implement module-level _lagged_ols helper and filter wrapper that vetoes signals disagreeing with AR(p) forecast.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §4.1
- Files to modify/create: strategy/kairos_econometric.py (NEW), tests/unit/test_econometric.py
- Relevant existing code: None yet (new module)
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Module-level `_lagged_ols(closes, p)` helper for OLS on lags (statsmodels-free)
- ARIMADisagreementStrategy filter wrapper fits AR(p) with drift on trailing 120 closes
- AIC selects p from range 1..5, verified in `test_arima_aic_selection`
- Vetoes signal if ARIMA point forecast and Kronos mean forecast disagree in direction, verified in `test_arima_disagreement_veto`
- Boosts confidence by agree_boost (default 1.2, capped at 1.0) when they agree, verified in `test_arima_agreement_boost`
- On pure trend series AR forecast sign matches trend direction, verified in `test_arima_trend_detection`

## Definition of done
- Unit tests in `tests/unit/test_econometric.py` pass via `uv run --with pytest python -m pytest tests/unit/test_econometric.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
