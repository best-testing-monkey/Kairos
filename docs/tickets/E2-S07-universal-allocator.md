# E2-S07: Universal Allocator

## Goal
Implement Cover's universal portfolio (wealth-weighted mixture of constant-rebalanced portfolios).

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.7
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- UniversalAllocator class as stateful allocator (must expose reset() method per APPENDIX-A standards)
- Maintains Dirichlet grid of constant-rebalanced portfolios over signaled assets (resolution 0.1)
- Tracks cumulative wealth per grid point, outputs wealth-weighted mixture, verified in `test_universal_wealth_weighted_output`
- On synthetic data where one asset dominates, weights converge toward it, verified in `test_universal_convergence_to_dominant`
- Total weight always sums to 1, verified in `test_universal_weights_sum_to_one`
- Grid regenerated when universe changes, verified in `test_universal_grid_regeneration`
- reset() method clears state correctly, verified in `test_universal_reset_clears_state`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
