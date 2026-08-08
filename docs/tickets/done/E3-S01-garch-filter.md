# E3-S01: GARCH Filter

## Goal
Implement GARCH(1,1) filter wrapper that blocks signals during high-volatility regimes.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §3.1
- Files to modify/create: strategy/kairos_volatility.py, tests/unit/test_volatility.py
- Relevant existing code: KurtosisFilterStrategy pattern in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- GARCH(1,1) filter wrapper using scipy.optimize (no arch dependency), 3 params, L-BFGS-B, variance targeting for ω
- Fits on trailing 250 log returns, forecasts next-day σ
- Blocks wrapped strategy's signal when forecast σ exceeds sigma_cap (default: 90th percentile of trailing fitted σ)
- Analogous to KurtosisFilterStrategy blocking behavior, verified in `test_garch_blocks_on_high_vol`
- On simulated GARCH data recovers alpha+beta within ±0.1, verified in `test_garch_fits_simulated_data`
- Refitted weekly, cached otherwise per APPENDIX-A standards
- Falls back to pass-through with warning if MLE fails to converge, verified in `test_garch_fallback_on_convergence_failure`

## Definition of done
- Unit tests in `tests/unit/test_volatility.py` pass via `uv run --with pytest python -m pytest tests/unit/test_volatility.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
