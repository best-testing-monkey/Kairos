# E7-S05: TCA Report

## Goal
Implement post-backtest transaction cost analysis decomposing slippage into timing and impact components.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.3
- Files to modify/create: strategy/kairos_execution.py, strategy/kairos_backtest.py, tests/unit/test_execution_algos.py
- Relevant existing code: compute_metrics() function from strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `compute_tca(trades) -> DataFrame` function (not a strategy) analyzing per-trade slippage
- Decomposes slippage into: timing component (entry vs day open) and impact component (assumed bps)
- Columns sum to total slippage per trade, verified in `test_tca_columns_sum_to_total`
- Empty trade list handled gracefully (returns empty DataFrame, no exception), verified in `test_tca_empty_trades`
- Wired into compute_metrics output as optional section
- Per-trade slippage components clearly labeled

## Definition of done
- Unit tests in `tests/unit/test_execution_algos.py` pass via `uv run --with pytest python -m pytest tests/unit/test_execution_algos.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
