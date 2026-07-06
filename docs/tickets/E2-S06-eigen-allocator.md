# E2-S06: Eigen Allocator

## Goal
Implement PCA-based portfolio allocator excluding market mode (PC1) and weighting remaining eigenvectors.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.6
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- EigenAllocator class with `__init__(n_components=3, lookback=120)` constructor
- Performs PCA on correlation matrix, drops PC1 (market mode)
- Allocates to top-k remaining eigenvectors weighted by eigenvalue
- Projects back to asset space and re-signs by signal direction, verified in `test_eigen_pc1_exclusion_reduces_correlation`
- Eigen portfolios mutually orthogonal, verified in `test_eigen_portfolios_orthogonal`
- PC1 exclusion reduces average pairwise correlation of resulting weight vector with equal-weight basket, verified in `test_eigen_correlation_reduction`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
