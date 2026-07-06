# E4-S03: Seasonality Filter

## Goal
Implement filter wrapper detecting and vetoing signals fighting significant seasonal effects.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §4.3
- Files to modify/create: strategy/kairos_econometric.py, tests/unit/test_econometric.py
- Relevant existing code: None yet for STL-lite (seasonal decomposition)
- Standards: see docs/tickets/APPENDIX-A-standards.md, uses kairos.calendar for trading-day awareness

## Acceptance criteria
- SeasonalityFilterStrategy filter wrapper
- STL-lite decomposition: day-of-week and month-of-year mean effects estimated on trailing 2 years
- HAC-adjusted t-stats for significance testing
- Vetoes signals fighting significant (|t|>2) seasonal effect, verified in `test_seasonality_significant_veto`
- Passes signals aligned with or indifferent to seasonal effects, verified in `test_seasonality_aligned_pass`
- Uses kairos.calendar for trading-day awareness (not raw calendar days)
- On planted Friday effect in synthetic data, detects effect, verified in `test_seasonality_friday_effect_detection`
- No vetoes when all effects insignificant (t below threshold on white noise, 95% of runs), verified in `test_seasonality_white_noise_false_positive_rate`

## Definition of done
- Unit tests in `tests/unit/test_econometric.py` pass via `uv run --with pytest python -m pytest tests/unit/test_econometric.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
