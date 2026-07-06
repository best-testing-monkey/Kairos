# E1-S01: Walk-Forward Harness

## Goal
Implement a walk-forward cross-validation framework that rolls anchored or sliding windows, avoids state leakage, and quantifies overfitting via in-sample vs out-of-sample Sharpe degradation.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §7.1
- Files to modify/create: strategy/kairos_backtest.py, tests/unit/test_walk_forward.py
- Relevant existing code: `backtest()` function and `compute_metrics()` in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `walk_forward()` function accepts `strategy_factory`, `data`, `train_days=250`, `test_days=60`, `step=60` parameters
- Folds never overlap in test data; all data partitioning verified in `test_fold_partitioning_no_overlap`
- A deliberately overfit strategy (lookahead-peeking fixture) shows OOS Sharpe collapse and Deflated Sharpe Ratio < 0.5, verified in `test_overfit_strategy_dsr_below_threshold`
- Fixed-seed runs produce reproducible per-fold and aggregate metrics, verified in `test_walk_forward_reproducible`
- Overfitting score (DSR or equivalent Sharpe degradation) computed and returned with per-fold results

## Definition of done
- Unit tests in `tests/unit/test_walk_forward.py` pass via `uv run --with pytest python -m pytest tests/unit/test_walk_forward.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
