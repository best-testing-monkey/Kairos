# E11-S07: allocate() orchestration + cluster-map loading

## Goal
Add the top-level `allocate(candidates: list[Candidate], config: AllocationConfig, enabled_mask: dict = None) -> AllocationResult` entry point to `strategy/allocation.py`, wiring together E11-S01 through E11-S06 into the single pure reference implementation RFC §8 calls "the reference oracle", plus a `load_cluster_map(path) -> dict` helper for `config.cluster_map`.

## Context
- Read: docs/rfc_allocation_sheet.md §4 intro ("The Python allocate() implementation is kept as the reference oracle..."), §3.1 (`cluster_map` config entry: "ticker -> cluster name, static mapping" from a file)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation.py (append)
- Depends on: E11-S01 (fetch_signals/Candidate), E11-S02 (validate_candidate), E11-S03 (AllocationConfig/compute_derived), E11-S04 (compute_ev_ratio), E11-S05 (select_candidates), E11-S06 (size_selected)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `AllocationResult` is a dataclass with at least: `rows: list[dict]` (one per input candidate, each carrying its final `status`, `flags`, `alloc`, and all derived fields), `selected_count: int`, `gross_exposure_pct: float` (sum of `alloc` across `status == "SELECTED"` rows), `rejection_counts: dict[str, int]` (status → count, for all non-selected rows)
- `allocate()` runs candidates through: `validate_candidate` (E11-S02) → `compute_derived` (E11-S03) → `compute_ev_ratio` (E11-S04) → `select_candidates` (E11-S05) → `size_selected` (E11-S06), in that order, and assembles `AllocationResult`
- `enabled_mask` defaults to all-enabled (empty dict, since `select_candidates`'s `DISABLED` check already defaults missing tickers to enabled per E11-S05)
- `load_cluster_map(path)` reads a simple two-column format (ticker, cluster — CSV, since no other format is specified in the RFC; document the chosen format in a docstring) and returns a `dict[str, str]`; missing tickers in the map should NOT crash `size_selected` — treat an unmapped ticker as its own singleton cluster (its own ticker name as cluster key) so cluster-cap logic degrades gracefully
- Determinism test: calling `allocate()` twice with the same candidates/config/mask produces byte-identical `AllocationResult.rows` (same order, same values) — RFC §4.4's "Same input file plus same Enabled mask must always produce the same output"
- End-to-end test using the RFC §7 worked-example rows (NG=F close_direction, NG=F open_gap, REMX, V, CRM) — reproduce the RFC's described outcomes: NG=F close_direction wins the DUP_ASSET collapse over NG=F open_gap, REMX carries `DATA_MISMATCH`, and relative Score/Kelly ordering matches §7's table (score ranks NG=F over REMX despite REMX's higher raw EV)

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
