# E8-S01: Multi-Factor Rank

## Goal
Implement cross-sectional multi-factor composite strategy extending CrossAssetRankStrategy.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.4
- Files to modify/create: strategy/kairos_meta.py, tests/unit/test_factor_strategies.py
- Relevant existing code: CrossAssetRankStrategy pattern in strategy/kairos_meta.py (single-factor rank)
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- MultiFactorRankStrategy cross-sectional composite over universe
- Factors: momentum (12-1 return), low-vol (inverse 60d σ), value proxy (distance below 252d high), quality proxy (return/σ stability)
- Z-scores each factor, averages into composite rank
- Longs top decile / shorts bottom (or top-1/bottom-1 for 3-asset mode)
- Gated on Kronos agreement
- Z-scoring and composite ranks verified on fixture, verified in `test_multi_factor_z_scoring`
- Degenerates to CrossAssetRankStrategy behavior with momentum-only weights (1.0/0/0/0), verified in `test_multi_factor_momentum_only_degenerates`
- Factors contribute equally when weighted uniformly

## Definition of done
- Unit tests in `tests/unit/test_factor_strategies.py` pass via `uv run --with pytest python -m pytest tests/unit/test_factor_strategies.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
