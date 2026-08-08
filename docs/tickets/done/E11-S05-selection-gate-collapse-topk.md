# E11-S05: Selection — gate, per-asset collapse, top-K

## Goal
Add `select_candidates(candidates: list[Candidate], config: AllocationConfig, enabled_mask: dict) -> list[dict]` to `strategy/allocation.py`, implementing the first three stages of RFC §4.4 (gate, collapse, rank+top-K) with the rejection-reason enum from §4.5.

## Context
- Read: docs/rfc_allocation_sheet.md §4.4 (gate / collapse / rank+top-K blocks only — sizing/caps/dust are E11-S06), §4.5 (rejection enum)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation.py (append)
- Depends on: E11-S01 (`Candidate`), E11-S02 (`validate_candidate` for SCHEMA_ERROR), E11-S03 (`compute_derived` for `ev_net`/`score`), E11-S04 (`compute_ev_ratio` for the `DATA_MISMATCH` flag carried through, informational only — does not reject)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Each candidate becomes a row-dict carrying: the original `Candidate` fields, `derived` fields (from E11-S03), a `status` field, and a `flags` list (starts with `["DATA_MISMATCH"]` if E11-S04 flagged it, else `[]`)
- Gate order and reasons exactly as RFC §4.4: `SCHEMA_ERROR` (from E11-S02) → `DISABLED` (`enabled_mask.get(ticker, True) is False`) → `LOW_N` (`n < config.min_n`) → `NEG_EV_NET` (`ev_net <= 0`) — first matching reason wins, checked in this order
- Per-asset collapse: group survivors by `ticker`; if both long and short directions survive gating for the same ticker → all rows for that ticker get `DIRECTION_CONFLICT`; else keep the max-`score` row, reject the rest as `DUP_ASSET`
- Rank + top-K: sort remaining survivors by `score` descending, tie-break by insertion order (stable sort — do NOT re-sort by `n`/ticker; RFC §4.4's tie-break note is about the *sheet's* pre-sorted row order, not this pure-Python reference, so a stable sort on the input order given is sufficient here); top `config.top_k` get no rejection yet (final `status` is assigned later in E11-S06), rest get `BELOW_TOPK`
- Returns ALL candidates (selected and rejected), each with its `status` set to its rejection reason, or `None`/unset for the ones proceeding to E11-S06's sizing stage
- Tests: one per gate reason, one for direction-conflict, one for dup-asset collapse keeping the higher-score row, one for below-topk, and one happy-path multi-ticker scenario checking final survivor set and order

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
