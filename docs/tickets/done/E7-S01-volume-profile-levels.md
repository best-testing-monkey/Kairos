# E7-S01: Volume Profile Levels

## Goal
Implement volume profile-based signal bracket adjustment snapping stops/targets to support/resistance nodes.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.2
- Files to modify/create: strategy/kairos_execution.py, tests/unit/test_execution_algos.py
- Relevant existing code: SupportConfluenceStrategy pattern in strategy/kairos_backtest.py
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- VolumeProfileLevelsStrategy wrapper
- Builds 60-day volume-at-price histogram (20 bins between rolling min/max)
- Identifies POC (point of control), VAH (value area high), VAL (value area low)
- Snaps wrapped signal's stop to nearest high-volume node (support), verified in `test_volume_profile_stop_snap`
- Snaps target to nearest low-volume node (target through the gap), verified in `test_volume_profile_target_snap`
- POC/VAH/VAL computed correctly on fixture, verified in `test_volume_profile_poc_vah_val`
- Stop only moves to a nearer level (never further from entry), verified in `test_volume_profile_stop_only_tightens`

## Definition of done
- Unit tests in `tests/unit/test_execution_algos.py` pass via `uv run --with pytest python -m pytest tests/unit/test_execution_algos.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
