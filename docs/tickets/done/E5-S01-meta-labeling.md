# E5-S01: Meta-Labeling

## Goal
Implement flagship meta-labeling strategy with triple-barrier labeling and logistic regression secondary classifier.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §5.1
- Files to modify/create: strategy/kairos_ml.py (NEW), tests/unit/test_ml_strategies.py
- Relevant existing code: Signal dataclass from strategy/kairos_backtest.py with bracket fields (stop/target)
- Standards: see docs/tickets/APPENDIX-A-standards.md (stateful, must expose reset())

## Acceptance criteria
- MetaLabelStrategy filter wrapper (stateful, must expose reset() per APPENDIX-A standards)
- Triple-barrier labeling of base-strategy signals (profit-take/stop/time via signal bracket)
- Secondary classifier trained on: entropy, kurtosis, skew, CDF position, ATR ratio, trailing strategy hit-rate, regime id
- Numpy IRLS logistic regression (no sklearn dependency)
- Vetoes signals below p_min=0.55, verified in `test_meta_label_veto_below_threshold`
- Warm-up: pass-through for first 60 labeled signals, verified in `test_meta_label_warm_up_passthrough`
- On synthetic setup where signals win iff entropy < 2.0, achieves AUC > 0.9 after warm-up, verified in `test_meta_label_entropy_classifier_auc`
- reset() clears labeled history per APPENDIX-A standards, verified in `test_meta_label_reset_clears_history`

## Definition of done
- Unit tests in `tests/unit/test_ml_strategies.py` pass via `uv run --with pytest python -m pytest tests/unit/test_ml_strategies.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
