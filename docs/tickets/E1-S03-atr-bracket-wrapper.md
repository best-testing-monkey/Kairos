# E1-S03: ATR Bracket Wrapper

## Goal
Implement Wilder-smoothed ATR helper and ATRBracketStrategy wrapper that dynamically adjusts signal brackets based on volatility.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §3.4
- Files to modify/create: strategy/kairos_volatility.py (NEW), tests/unit/test_volatility.py
- Relevant existing code: DynamicBracketStrategy pattern in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Module-level `atr(history, n=14)` function computes Wilder-smoothed ATR
- ATR values match hand-computed/TA-Lib reference fixture values to 1e-6, verified in `test_atr_matches_reference`
- ATRBracketStrategy recomputes stop at entry ∓ k_stop*ATR(14) and target at entry ± k_target*ATR(14), verified in `test_atr_bracket_computation`
- Stop only ever tightens, never widens (compared to original signal stop), verified in `test_atr_stop_only_tightens`
- Direction-consistent: stop below entry for longs, above for shorts, verified in `test_atr_stop_direction_consistency`

## Definition of done
- Unit tests in `tests/unit/test_volatility.py` pass via `uv run --with pytest python -m pytest tests/unit/test_volatility.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
