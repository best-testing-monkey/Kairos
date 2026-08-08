# E11-S10: write_ods_sheet()

## Goal
Add `write_ods_sheet(document, result: AllocationResult, config: AllocationConfig, report_date, generator_version: str)` to `strategy/allocation.py`, writing the same `Allocation` sheet structure as E11-S09 into an existing `odfpy` document, using `of:=`-dialect formulas.

## Context
- Read: docs/rfc_allocation_sheet.md §5 (same layout as E11-S09 — identical structure in both formats per §5 intro), §4.6 last paragraph ("ODS protection support in odfpy is unreliable, so ODS ships unprotected with a header note (accepted risk)")
- Files to modify: strategy/allocation.py; tests/unit/test_allocation_ods.py (new)
- Depends on: E11-S07 (`AllocationResult`), E11-S08 (`render_formula(..., fmt="ods")`), E11-S09 (mirror its layout exactly — same row/column plan, different writer API)
- `odfpy` is already a project dependency (added for `strategy/kairos_signals.py`'s `write_spreadsheet`) — reuse the same import convention, no new dependency needed
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Sheet named exactly `Allocation`, identical row/column layout to E11-S09 (title row, config block, summary block, instruction line, header row, data rows, cluster-exposure block)
- Formula cells use `render_formula(..., fmt="ods")` from E11-S08 (of:=, `;` separators) — no formula string duplicated between this file and E11-S09
- No sheet protection applied (odfpy protection is unreliable per RFC) — instead, a visible header note/cell stating the sheet is unprotected (RFC §4.6's "accepted risk" language, surfaced to the user)
- Test: write to an in-memory/temp `.ods` document via odfpy, reload it, assert sheet name, header row values match E11-S09's header row values exactly (same columns, same order), and at least one formula cell's content string matches `render_formula`'s ods output for that cell

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation_ods.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
