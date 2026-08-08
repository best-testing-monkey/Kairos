# E4-S06: Matrix Profile Anomaly

## Goal
Implement STOMP-based matrix profile anomaly detection and motif-following strategy.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §4.6
- Files to modify/create: strategy/kairos_econometric.py, tests/unit/test_econometric.py
- Relevant existing code: None yet (pure numpy implementation, no stumpy dependency)
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- MatrixProfileAnomalyStrategy standalone strategy using STOMP over trailing 250 closes
- Window length 20, z-normalized, pure numpy implementation (no stumpy dependency)
- Discord detection: today's window is discord when profile value > mean + 2σ, triggers abstention
- Strong motif match: trades direction that followed historical match, gated on Kronos agreement, sized by match quality
- Matrix profile matches stumpy reference implementation on fixture within 1e-4, verified in `test_matrix_profile_matches_stumpy`
- Planted repeated motif is found, verified in `test_matrix_profile_finds_motif`
- Discord abstention fires on planted anomaly, verified in `test_matrix_profile_discord_abstention`

## Definition of done
- Unit tests in `tests/unit/test_econometric.py` pass via `uv run --with pytest python -m pytest tests/unit/test_econometric.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
