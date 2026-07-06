# E2-S09: CVaR Allocator

## Goal
Implement Conditional Value-at-Risk (CVaR) portfolio optimization using Kronos sample paths as scenario set.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.9
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- CVaRAllocator class using Rockafellar-Uryasev LP formulation
- Uses PRED_SAMPLES predicted returns per asset as scenario set (not historical returns)
- Minimizes CVaR_95 subject to w.mu >= target_return using scipy.optimize.linprog
- CVaR of chosen weights <= CVaR of equal weight on same scenario set, verified in `test_cvar_better_than_equal_weight`
- Infeasible target_return falls back to max-return vertex, verified in `test_cvar_infeasible_fallback`
- Respects signal direction and cap constraints

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
