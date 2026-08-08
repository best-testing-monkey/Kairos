# E2-S10: Kelly Allocator

## Goal
Implement continuous-time Kelly criterion allocator with fractional Kelly and shrinkage.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.10
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class and shrunk_covariance from E1-S02, KairosDistribution.kelly_fraction() from strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- KellyAllocator class computing w = f * Σ⁻¹ mu with mu from Kronos expected values
- Uses shrunk_covariance() from E1-S02 for robust inversion
- Fractional-Kelly f=0.25 default, clipped to cap constraints
- Single-asset case reduces to KairosDistribution.kelly_fraction() within tolerance, verified in `test_kelly_single_asset_matches_kelly_fraction`
- Doubling Σ halves weights, verified in `test_kelly_cov_doubling_halves_weights`
- Respects gross_cap and max_weight caps, verified in `test_kelly_respects_caps`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
