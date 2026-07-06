# E6-S02: ADX Gate

## Goal
Implement strategy-type router that gates execution based on trend strength (ADX).

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.1
- Files to modify/create: strategy/kairos_backtest.py, tests/unit/test_technical_filters.py
- Relevant existing code: ADX computation techniques
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- ADXGateStrategy wrapper routing by trend strength
- ADX(14) > 25 passes trend-type strategies, ADX < 20 passes mean-reversion-type
- Strategy type declared at wrap time via kind="trend"|"reversion"
- Blocks opposite type (blocks reversion-type when ADX>25, blocks trend-type when ADX<20)
- ADX computation matches fixture, verified in `test_adx_matches_fixture`
- Trend routing verified: trend kind passes only when ADX > 25, verified in `test_adx_gate_trend_routing`
- Reversion routing verified: reversion kind passes only when ADX < 20, verified in `test_adx_gate_reversion_routing`

## Definition of done
- Unit tests in `tests/unit/test_technical_filters.py` pass via `uv run --with pytest python -m pytest tests/unit/test_technical_filters.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
