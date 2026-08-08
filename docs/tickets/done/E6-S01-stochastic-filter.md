# E6-S01: Stochastic Filter

## Goal
Implement Stochastic Oscillator filter wrapper following RSIFilterStrategy pattern.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.1
- Files to modify/create: strategy/kairos_backtest.py, tests/unit/test_technical_filters.py
- Relevant existing code: RSIFilterStrategy pattern in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- StochasticFilterStrategy filter wrapper with %K(14)/%D(3) computation
- Follows RSIFilterStrategy pattern (not a duplicate structure), verified via code review
- Vetoes longs when %K>80 unless trending (ADX>25), verified in `test_stochastic_overbought_long_veto`
- Vetoes shorts when %K<20 unless trending (ADX>25), verified in `test_stochastic_oversold_short_veto`
- %K/%D values match TA-Lib fixture, verified in `test_stochastic_matches_talib`
- Veto logic truth-table tested: all 4 combinations of overbought/oversold x trending/not, verified in `test_stochastic_veto_truth_table`

## Definition of done
- Unit tests in `tests/unit/test_technical_filters.py` pass via `uv run --with pytest python -m pytest tests/unit/test_technical_filters.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
