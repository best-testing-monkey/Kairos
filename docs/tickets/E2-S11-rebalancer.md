# E2-S11: Rebalancer

## Goal
Implement Rebalancer class that converts allocator target weights into trade deltas with threshold-based or periodic rebalancing.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §7.2
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Rebalancer class with `__init__(allocator, mode="threshold", band=0.05, min_interval_days=5)` constructor
- Converts allocator target weights into trade deltas
- No trades within band (threshold mode), verified in `test_rebalancer_no_trades_within_band`
- Respects min_interval_days in periodic mode, verified in `test_rebalancer_respects_min_interval`
- Respects per-trade transaction cost from backtest config so turnover is penalized
- Turnover under threshold mode < turnover under daily full rebalance on same weight stream, verified in `test_rebalancer_turnover_reduced`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
