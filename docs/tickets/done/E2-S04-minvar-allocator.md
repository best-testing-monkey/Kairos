# E2-S04: MinVar Allocator

## Goal
Implement minimum-variance portfolio allocation with shrunk covariance handling for singular matrices.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.5
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class and shrunk_covariance from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- MinVarAllocator class minimizing w'Σw using shrunk_covariance() from E1-S02
- Uses same caps as MVO (gross_cap, max_weight from §2.1), verified in `test_minvar_respects_caps`
- Output covariance is positive definite for n_assets > n_obs, verified in `test_minvar_pd_with_shrinkage`
- Weights respect gross_cap and max_weight, verified in `test_minvar_cap_enforcement`
- Uses shrunk_covariance helper (not a duplicate implementation), verified via code review

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
