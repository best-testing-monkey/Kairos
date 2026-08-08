# E11-S11: write_md_section()

## Goal
Add `write_md_section(result: AllocationResult, config: AllocationConfig) -> str` to `strategy/allocation.py`, rendering the RFC §6 Markdown "Portfolio Allocation" section as a static snapshot of the all-enabled, default-config `AllocationResult`.

## Context
- Read: docs/rfc_allocation_sheet.md §6 (exact section format, including the example table and rejection-count summary line)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation_md.py (new)
- Depends on: E11-S07 (`AllocationResult` — this writer consumes `allocate()`'s output directly, no formulas, per RFC §4's "Markdown writer consumes allocate() output directly")
- Reuse `strategy/kairos_signals.py`'s `format_table()` (kairos_signals.py:157-212) for the markdown table rendering — do not reimplement table formatting
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Output starts with `## Portfolio Allocation` heading
- A config summary line listing `n0`, `min_n`, `round_trip_cost_pct` (labeled "cost"), `kelly_mult`, `top_k`, `max_pos_pct`, `max_cluster_pct`, `gross_cap_pct` values from the `AllocationConfig` passed in, format matching RFC §6's example line
- A "Selected N of M signals. Gross exposure: X.X%." line using `result.selected_count`, `len(result.rows)`, `result.gross_exposure_pct`
- A markdown table (via `format_table`) with columns Ticker, Dir, Strategy, Entry, Stop, Target, EV net, Score, Alloc — one row per `status == "SELECTED"` row, sorted by score descending (already sorted per E11-S07's ordering — do not re-sort)
- A "Cluster exposure: ..." line listing each cluster's summed alloc among selected rows, comma-separated, format matching RFC §6's example
- A "Rejected: N total -- REASON count, REASON count, ..." line built from `result.rejection_counts`, sorted by count descending (matches RFC §6's example ordering: DUP_ASSET, BELOW_TOPK, LOW_N, NEG_EV_NET, DIRECTION_CONFLICT)
- Test: build a small `AllocationResult` by hand (or via `allocate()` on a small fixture), assert every required substring/line appears in the rendered output in the right order

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation_md.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
