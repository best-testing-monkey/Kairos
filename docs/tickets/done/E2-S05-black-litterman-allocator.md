# E2-S05: Black-Litterman Allocator

## Goal
Implement Black-Litterman model combining equilibrium prior with Kronos-derived views on expected returns.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.4
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: MVO optimizer from E2-S01, PortfolioAllocator base from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- BlackLittermanAllocator class with `__init__(tau=0.05, delta=2.5, lookback=120)` constructor
- Prior equilibrium returns Π = δ Σ w_mkt with w_mkt = inverse-vol weights (no market-cap needed)
- Views: one absolute view per signaled asset, Q_i = dists[i].stats["close"]["mean"]/price - 1
- View uncertainty Ω_ii ∝ dists[i].entropy(), verified in `test_bl_view_uncertainty_from_entropy`
- With zero-confidence views (Ω→∞) output equals prior weights, verified in `test_bl_zero_confidence_equals_prior`
- With infinite-confidence views output matches MVO on Q, verified in `test_bl_infinite_confidence_matches_mvo`
- Entropy=ln(20) view moves posterior <10% of the way from prior to view, verified in `test_bl_max_entropy_weak_view`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
