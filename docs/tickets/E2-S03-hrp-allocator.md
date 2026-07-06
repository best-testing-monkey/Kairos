# E2-S03: HRP Allocator

## Goal
Implement Hierarchical Risk Parity (Lopez de Prado) using correlation-distance clustering and quasi-diagonalization.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.3
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- HRPAllocator class with correlation-distance matrix construction
- Uses scipy.cluster.hierarchy.linkage (single) for clustering
- Implements quasi-diagonalization and recursive bisection with inverse-variance splits
- No matrix inversion required (uses hierarchical approach instead)
- On a fixed 4-asset synthetic covariance defined in the test (seeded, block-structured: two correlated pairs with vols 0.1/0.1/0.2/0.2), weights are deterministic, sum to 1, and the low-vol pair receives more total weight than the high-vol pair, verified in `test_hrp_synthetic_covariance`
- Handles n_assets=2 (degenerates to inverse-variance), verified in `test_hrp_2asset_degenerates_to_inv_var`
- Supports `variant="herc"` for cluster-level equal risk contribution, verified in `test_hrp_herc_variant`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
