# E3-S03: Variance Risk Premium

## Goal
Implement standalone strategy trading variance risk premium based on Kronos-implied vs realized volatility.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §3.3
- Files to modify/create: strategy/kairos_volatility.py, tests/unit/test_volatility.py
- Relevant existing code: TailAsymmetryStrategy and RangeTradingStrategy patterns in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- VarianceRiskPremiumStrategy standalone strategy
- Computes Kronos-implied variance from prediction sample dispersion vs trailing 20-day realized variance
- Entry condition: implied >> realized (ratio > entry_ratio, default 1.5) for vol expansion expectation
- Enters in TailAsymmetryStrategy-style direction with wide bracket (stop pct_5/target pct_95), verified in `test_vrp_expansion_entry`
- Compression condition: implied << realized (ratio < 1/entry_ratio) for vol compression expectation
- Enters RangeTradingStrategy-style fade with tight bracket, verified in `test_vrp_compression_entry`
- No signal when ratio in [1/entry_ratio, entry_ratio], verified in `test_vrp_no_signal_near_parity`
- Bracket widths verified against distribution percentiles, verified in `test_vrp_bracket_widths`

## Definition of done
- Unit tests in `tests/unit/test_volatility.py` pass via `uv run --with pytest python -m pytest tests/unit/test_volatility.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
