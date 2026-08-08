# E11-S06: Sizing — Kelly cap, cluster caps, gross cap, dust filter

## Goal
Add `size_selected(survivors: list[dict], config: AllocationConfig) -> list[dict]` to `strategy/allocation.py`, implementing the sizing tail of RFC §4.4 (position cap, cluster caps, gross cap, dust filter) applied to the rows E11-S05 marked as passing top-K.

## Context
- Read: docs/rfc_allocation_sheet.md §4.4 (sizing block onward — position cap through dust filter), §4.5 (informational flags `CLUSTER_CAPPED`, `POS_CAPPED`), §4.6's "Deliberate simplifications" subsection (dust is single-pass, not redistributed — this applies to the Python reference too, per that same paragraph)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation.py (append)
- Depends on: E11-S05 (row-dicts with `status` unset for top-K survivors, `derived["kelly_frac"]` from E11-S03, `config.cluster_map` from E11-S03 for looking up each ticker's cluster)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- For each top-K survivor: `alloc_raw = min(kelly_frac * 100, config.max_pos_pct)`; if `kelly_frac * 100 > config.max_pos_pct`, append `POS_CAPPED` to that row's `flags`
- Cluster caps: group top-K survivors by `config.cluster_map[ticker]`; if a cluster's summed `alloc_raw` exceeds `config.max_cluster_pct`, scale that cluster's allocations down proportionally so the sum equals the cap exactly, and append `CLUSTER_CAPPED` to each scaled row's `flags`
- Gross cap: if the sum of all (post-cluster-cap) allocations exceeds `config.gross_cap_pct`, scale ALL allocations down proportionally so the sum equals the cap exactly (single global scale factor, not per-cluster)
- Dust filter: any row whose final allocation is `< config.dust_min_pct` gets `alloc = 0` and `status = "DUST"` — single pass, no redistribution of freed capital to other rows (confirm no second pass recomputes sums/caps after dust removal)
- Rows that reach the end without being zeroed get `status = "SELECTED"`
- Returns the full row set (selected + dust-zeroed), each with final `alloc` and `status` set
- Tests: no caps triggered (happy path), position cap triggered (`POS_CAPPED` flag + value clamped), cluster cap triggered (proportional scale-down verified numerically, `CLUSTER_CAPPED` flag), gross cap triggered (all rows scaled by the same factor), dust filter zeroing a small allocation, and a combined scenario exercising cluster cap → gross cap → dust in sequence to confirm the cascade order matches RFC §4.4

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
