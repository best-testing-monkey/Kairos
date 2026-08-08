# E2-S01: MVO Allocator

## Goal
Implement Markowitz maximum-Sharpe portfolio optimization using Kronos-derived expected returns.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.1
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- MVOAllocator class with `__init__(lookback=120, gross_cap=1.0, max_weight=0.35, rf=0.0)` constructor
- With two uncorrelated assets of equal mu, weights split ~50/50, verified in `test_mvo_equal_mu_splits_50_50`
- Raising one asset's mu monotonically raises its weight, verified in `test_mvo_monotonic_mu_weight`
- Never violates gross_cap or max_weight caps, verified in `test_mvo_respects_caps`
- Uses scipy.optimize.minimize with method="SLSQP" to maximize (w.mu - rf)/sqrt(w'Σw)

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
