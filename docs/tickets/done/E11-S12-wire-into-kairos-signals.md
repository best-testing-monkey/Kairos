# E11-S12: Wire allocation.py into kairos_signals.py's report output

## Goal
Call the allocation pipeline from `strategy/kairos_signals.py`'s `run()`, appending the RFC §6 Markdown section to the report and adding the `Allocation` sheet to the existing `--xlsx`/`--ods` outputs (no new CLI flags needed — allocation runs whenever a report is generated).

## Context
- Read: docs/rfc_allocation_sheet.md §8 (module boundaries — allocation.py is a separate module, kairos_signals.py just calls into it)
- Files to modify: strategy/kairos_signals.py (in `run()`, after `stats_rows`/`advice_rows` are fully built, before the markdown/xlsx/ods writing block); tests/unit/test_signals_report.py (append)
- Depends on: E11-S01 (`fetch_signals`), E11-S07 (`allocate`, `AllocationConfig`, `load_cluster_map`), E11-S09 (`write_xlsx_sheet`), E11-S10 (`write_ods_sheet`), E11-S11 (`write_md_section`)
- Exact insertion points in kairos_signals.py: markdown is written at kairos_signals.py:701-707 (`report = render_report(...)`, `f.write(report)`) — append `write_md_section(...)` output before writing to `f`. XLSX/ODS is written in `write_spreadsheet()` (kairos_signals.py:463-488) inside the `with pd.ExcelWriter(out_path, engine=engine) as writer:` block (line 485) — the underlying workbook/document object is available as `writer.book` regardless of engine (`openpyxl.Workbook` for `engine="openpyxl"`, odfpy document for `engine="odf"`); call `write_xlsx_sheet(writer.book, ...)` or `write_ods_sheet(writer.book, ...)` there, inside the same `with` block, so the Allocation sheet lands in the same file as strategies/signals
- `AllocationConfig`'s `cluster_map` needs a source file path — add a `--cluster_map` CLI flag (optional, default `None` meaning empty map) to `main()`, forwarded through `run()`, loaded via `load_cluster_map` (E11-S07) if given
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `run()` gains an `AllocationConfig`-shaped set of parameters (either one `allocation_config: AllocationConfig = None` param defaulting to `AllocationConfig()`, or a `cluster_map_path: str = None` param — implementer's call on which is cleaner given the existing `run()` signature's style) — do not change any existing `run()` parameter's meaning or default
- After `stats_rows`/`advice_rows` are built (existing loop, kairos_signals.py:583-694 unchanged), call `fetch_signals(stats_rows, advice_rows)` then `allocate(candidates, config)`, guarding for the empty-candidates case (no signals at all → skip the allocation section entirely, matching the existing `_No signals generated._` fallback pattern for consistency)
- Markdown report gets `write_md_section(...)`'s output appended after the existing report body, before the Legend/Failures/Skipped sections OR after them — match RFC §6's stated position ("Appended after the existing signal list")
- `--xlsx`/`--ods` outputs (when those flags are set) gain the `Allocation` sheet alongside the existing `strategies`/`signals` sheets, via the `writer.book` access pattern described in Context
- New `--cluster_map PATH` CLI flag, optional, help text explains it's the ticker->cluster static mapping file for the Allocation sheet
- Existing behavior when there are zero candidates (empty stats_rows) is unchanged — no crash, no empty Allocation sheet/section
- Test: run the full `run()` flow with a small mocked `predict_fn`/`fetch_data_raw` (reuse the existing test pattern from `TestRenderReport`/similar classes in test_signals_report.py) producing a couple of signals, assert the markdown output contains `## Portfolio Allocation`, and assert `--xlsx` output (via `write_spreadsheet`) contains an `Allocation` sheet name when read back

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_signals_report.py -q` passes
- `uv run --with pytest python -m pytest tests/unit/ -q` full suite passes (no regressions to existing kairos_signals.py behavior)
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
