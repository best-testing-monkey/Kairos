# E4-S05: Granger Pairs

## Goal
Implement standalone strategy detecting Granger causality between asset pairs and trading cross-asset implications.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §4.5
- Files to modify/create: strategy/kairos_econometric.py, tests/unit/test_econometric.py
- Relevant existing code: _lagged_ols helper from E4-S01
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- GrangerPairsStrategy standalone strategy
- Rolling Granger F-test (lags 1-3) between all asset pairs
- Trades the follower in direction implied by leader's yesterday move times fitted coefficient sign
- Gated on Kronos agreement for directional confirmation
- Shares OLS machinery with E4-S02 (_lagged_ols helper), verified via code review
- F-test p-values match statsmodels reference on fixture within 1e-4, verified in `test_granger_matches_statsmodels`
- Symmetric independence produces no signals, verified in `test_granger_independent_pairs_no_signal`

## Definition of done
- Unit tests in `tests/unit/test_econometric.py` pass via `uv run --with pytest python -m pytest tests/unit/test_econometric.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
