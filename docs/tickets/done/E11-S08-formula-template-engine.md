# E11-S08: Formula template engine (XLSX/ODS dialect rendering)

## Goal
Add a small formula-templating layer to `strategy/allocation.py` (or a new `strategy/allocation_formulas.py` if that keeps the module cleaner — implementer's call, but keep it one file) that stores each RFC §4.6 formula ONCE and renders it for either dialect: A1 strings for openpyxl (XLSX), or `of:=` namespaced with `;` argument separators for odfpy (ODS).

## Context
- Read: docs/rfc_allocation_sheet.md §4.6 in full (every formula O through AJ, plus the gross scale factor `$D$14`), §8 ("Formula templates defined once, with placeholders for ranges and config cells, rendered per dialect... No formula string is ever written twice by hand.")
- Files to modify: strategy/allocation.py (or new strategy/allocation_formulas.py); tests/unit/test_allocation_formulas.py (new)
- This story does NOT write to actual XLSX/ODS files (that's E11-S09/S10) — it only produces formula strings for a given row number and format
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- One template per formula (O, P, Q, R, S, T, U, V, W, X, Y, Z, AA, AB, AC, AD, AE, AF, AG, AH, AI, AJ, and the summary-block gross scale factor), each written once, each renderable for a given data row number (rows 20..N per RFC §5.1) and format (`"xlsx"` or `"ods"`)
- `render_formula(name: str, row: int, fmt: str) -> str` (or equivalent) returns:
  - for `fmt="xlsx"`: the exact A1-style formula string as shown in RFC §4.6 (comma-separated arguments, `=` prefix)
  - for `fmt="ods"`: the same logical formula with `of:=` prefix and `;` as the argument separator instead of `,` (per RFC §4.6's dialect note), same cell references
- Both dialects reference the SAME underlying formula definition (single source per formula name) — verify this structurally in a test, e.g. by asserting both dialects' outputs derive from one shared template table, not two independently-written formula sets
- Only the compatibility-subset functions from RFC §4 are used: `IF, AND, OR, NOT, MIN, MAX, SUM, ABS, ROW, SUMPRODUCT, SUMIFS, COUNTIFS, IFERROR` — a test scans all rendered formulas (any row) and asserts no banned function name (`FILTER`, `SORT`, `UNIQUE`, `LET`, `LAMBDA`, `XLOOKUP`, `MAXIFS`, `MINIFS`) appears as a substring
- Row-number substitution is correct for at least rows 20, 21, and 400 (boundary of the `A$20:A$400` ranges) — ranges themselves stay fixed at `$20:$400` while the row-relative references (e.g. `F20`, `E20`) update per row

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation_formulas.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
