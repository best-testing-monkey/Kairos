# E5-S03: LPPLS Guard

## Goal
Implement log-periodic power law singularity (LPPLS) bubble detector via nonlinear Nelder-Mead optimization.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §5.3
- Files to modify/create: strategy/kairos_ml.py, tests/unit/test_ml_strategies.py
- Relevant existing code: None yet (nonlinear optimization from scipy.optimize)
- Standards: see docs/tickets/APPENDIX-A-standards.md (must cache fit and refit weekly)

## Acceptance criteria
- LPPLSGuardStrategy filter wrapper
- Fits log-periodic power law singularity model (Sornette) on trailing 250 log-prices
- Nonlinear fit of (tc, m, ω) with linear params profiled out (standard 3-param reduction)
- Multi-start Nelder-Mead optimization
- Bubble signature: fit quality passes Sornette filter (0.1 < m < 0.9, 6 < ω < 13, tc within 60 days)
- Vetoes new LONG entries and allows/boosts SHORT signals when bubble signature detected, verified in `test_lppls_bubble_blocks_longs`
- Flags synthetic super-exponential + log-periodic series, verified in `test_lppls_detects_bubble`
- Does not flag GBM paths (<10% false-positive rate over 100 seeds), verified in `test_lppls_gbm_false_positive_rate`
- Weekly caching of fits per APPENDIX-A standards

## Definition of done
- Unit tests in `tests/unit/test_ml_strategies.py` pass via `uv run --with pytest python -m pytest tests/unit/test_ml_strategies.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
