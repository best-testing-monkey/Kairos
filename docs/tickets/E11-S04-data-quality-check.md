# E11-S04: ev_implied vs ev_reported data-quality check

## Goal
Add `compute_ev_ratio(c: Candidate, derived: dict) -> tuple[float, bool]` to `strategy/allocation.py`, implementing RFC §4.3's `DATA_MISMATCH` flag.

## Context
- Read: docs/rfc_allocation_sheet.md §4.3 (exact formulas and the `[0.5, 2.0]` threshold)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation.py (append)
- Depends on: E11-S01 (`Candidate`), E11-S03 (`derived["reward_pct"]`/`derived["risk_pct"]`)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `ev_implied = base_win_rate * reward_pct - (1 - base_win_rate) * risk_pct` (uses `derived["reward_pct"]`/`derived["risk_pct"]` from E11-S03, not recomputed)
- `ev_ratio = ev_pct / ev_implied`, guarded against `ev_implied` near zero (e.g. `abs(ev_implied) < 1e-9` → treat as not-mismatched, since the ratio is undefined, not a data problem)
- Returns `(ev_ratio, is_mismatch)` where `is_mismatch` is `True` iff `ev_ratio` is outside `[0.5, 2.0]`
- Tests: a case inside the band (no mismatch), a case above 2.0, a case below 0.5, and the near-zero-`ev_implied` guard case

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
