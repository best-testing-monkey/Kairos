# E5-S02: GBM Direction

## Goal
Implement gradient-boosted tree classifier for next-day direction prediction with numpy implementation.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §5.2
- Files to modify/create: strategy/kairos_ml.py, tests/unit/test_ml_strategies.py
- Relevant existing code: None yet (pure numpy gradient boosting, no sklearn)
- Standards: see docs/tickets/APPENDIX-A-standards.md (must cache fit and refit weekly)

## Acceptance criteria
- GBMDirectionStrategy standalone strategy
- Numpy implementation: 50 trees, depth 2, learning rate 0.1, logloss
- Features (~15 total): returns at 1/5/20d, RSI, ATR ratio, volume z-score, day-of-week, Kronos summary stats
- Trades only when classifier and Kronos agree AND P > 0.6, verified in `test_gbm_agreement_threshold`
- Retrains weekly on trailing 500 days, cached per APPENDIX-A standards, verified in `test_gbm_weekly_cache`
- Beats logistic baseline on synthetic nonlinear (XOR-of-features) data, verified in `test_gbm_beats_logistic_baseline`
- Deterministic given seed, verified in `test_gbm_deterministic_seed`

## Definition of done
- Unit tests in `tests/unit/test_ml_strategies.py` pass via `uv run --with pytest python -m pytest tests/unit/test_ml_strategies.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
