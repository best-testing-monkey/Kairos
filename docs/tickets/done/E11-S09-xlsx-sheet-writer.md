# E11-S09: write_xlsx_sheet()

## Goal
Add `write_xlsx_sheet(workbook, result: AllocationResult, config: AllocationConfig, report_date, generator_version: str)` to `strategy/allocation.py`, writing the `Allocation` sheet into an existing `openpyxl.Workbook` per RFC §5.

## Context
- Read: docs/rfc_allocation_sheet.md §5 (full: layout §5.1, unified table §5.2, cluster exposure §5.3, audit trail §5.4), §4.6 last paragraph (sheet protection: lock all cells except column N and config value cells)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation_xlsx.py (new)
- Depends on: E11-S07 (`AllocationResult`), E11-S08 (`render_formula` for the O-AJ formula columns)
- `openpyxl` is already a project dependency (added for `strategy/kairos_signals.py`'s `write_spreadsheet`) — reuse the same import convention, no new dependency needed
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Sheet named exactly `Allocation`
- Row 1: title + report date + generator version string (RFC §5.1)
- Rows 3-13: config block, 3 columns (parameter name, editable value, locked shipped-default value) — one row per `AllocationConfig` field (RFC §3.1's parameter list)
- Rows 14-16: summary block with live formulas for selected count, gross exposure %, gross scale factor (cell `$D$14` per RFC §4.6), enabled count
- Row 18: instruction line text (RFC §5.1)
- Row 19: header row matching RFC §5.2's column table (A through AJ, static columns A-N written as literal values from `AllocationResult.rows`, formula columns O-AJ rendered via `render_formula(..., fmt="xlsx")` from E11-S08 for each data row)
- Rows 20..N: one row per candidate in `result.rows`, pre-sorted by score descending (already sorted coming out of `allocate()` per E11-S05/S07 — do not re-sort here)
- A separate block below the data rows: Section B cluster-exposure table (RFC §5.3), one row per distinct cluster in `config.cluster_map` values, formulas per §5.3
- Sheet protection: all cells locked except column N (Enabled) and the editable config value column — use openpyxl's `Protection`/`sheet.protection.sheet = True` pattern
- Helper columns (whichever RFC §5.2 marks as "helper columns X-AF... grouped/collapsed") are grouped (openpyxl column grouping/outline), not hidden
- Test: write to an in-memory `openpyxl.Workbook`, reload it, assert sheet name, row/column layout at the cells RFC specifies (spot-check header row 19 values, a formula cell's string content matches `render_formula`'s xlsx output, config block values match `AllocationConfig` defaults, protection is enabled)

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation_xlsx.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
