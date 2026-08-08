# E1-S02: Allocator Base and Shrinkage

## Goal
Create a PortfolioAllocator base class with shrinkage covariance handling for singular or near-singular matrices.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2 intro + §2.5
- Files to modify/create: strategy/kairos_portfolio.py (NEW), tests/unit/test_portfolio.py
- Relevant existing code: None yet (new module)
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- PortfolioAllocator base class with `allocate(signals, returns, dists, context) -> Dict[str, float]` signature
- Shrinkage intensity in [0,1] verified in `test_shrinkage_intensity_bounds`
- With n=200 observations / 3 assets shrinkage intensity < 0.3, verified in `test_shrinkage_intensity_with_200_obs`
- Output covariance is positive definite for n_assets > n_obs, verified in `test_shrunk_covariance_positive_definite`
- Equal-weight fallback triggered when observations < min_obs=60, verified in `test_equal_weight_fallback_below_min_obs`
- Dependencies are numpy/scipy only (no cvxpy), verified via import inspection

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
