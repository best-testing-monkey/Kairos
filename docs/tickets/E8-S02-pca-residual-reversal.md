# E8-S02: PCA Residual Reversal

## Goal
Implement statistical factor model using PCA for residual mean reversion.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.4
- Files to modify/create: strategy/kairos_meta.py, tests/unit/test_factor_strategies.py
- Relevant existing code: None yet (PCA-based factor model)
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- PCAResidualReversalStrategy standalone strategy
- Performs PCA (k=1) on 60d returns (extracts first principal component)
- Computes each asset's residual vs factor reconstruction
- Fades assets with |residual z| > 2 back toward the factor
- Gated on Kronos agreement
- Residuals orthogonal to factor (dot product ~0), verified in `test_pca_residuals_orthogonal_to_factor`
- Reversal fires on planted idiosyncratic shock (one asset's residual pushed to z > 2), verified in `test_pca_reversal_fires_on_shock`

## Definition of done
- Unit tests in `tests/unit/test_factor_strategies.py` pass via `uv run --with pytest python -m pytest tests/unit/test_factor_strategies.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
