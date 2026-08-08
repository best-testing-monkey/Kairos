# E3-S02: Vol Target Sizer

## Goal
Implement volatility-targeting sizer that scales signal sizes based on blended realized and predicted volatility.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §3.2
- Files to modify/create: strategy/kairos_volatility.py, tests/unit/test_volatility.py
- Relevant existing code: GARCHFilterStrategy from E3-S01, ATRBracketStrategy from E1-S03
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- VolTargetSizer wrapper with target_vol default 15% annualized, max_leverage default 2.0
- Scales signal.size by target_vol / blended_vol
- Blended vol = 0.5 * (GARCH forecast) + 0.5 * (Kronos predicted range vol = (pct_84 - pct_16)/(2*price))
- Size halves when blended vol doubles, verified in `test_vol_target_sizer_halves_at_double_vol`
- Never exceeds base size * max_leverage, verified in `test_vol_target_sizer_respects_max_leverage`
- Never increases a zero-size signal, verified in `test_vol_target_sizer_zero_stays_zero`

## Definition of done
- Unit tests in `tests/unit/test_volatility.py` pass via `uv run --with pytest python -m pytest tests/unit/test_volatility.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
