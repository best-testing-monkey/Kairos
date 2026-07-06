# E2-S02: Risk Parity Allocator

## Goal
Implement equal risk contribution (ERC) allocator solving for weights where each asset contributes equally to portfolio risk.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.2
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- RiskParityAllocator class implementing Spinu formulation with SLSQP solver
- For 2 assets with vol 10%/20% and zero correlation, weights ≈ 2:1, verified in `test_risk_parity_2asset_vol_ratio`
- Risk contributions within 1% of each other at convergence, verified in `test_risk_parity_equal_contributions`
- Direction from signals, magnitude from ERC optimization
- Respects signal direction and cap constraints

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
