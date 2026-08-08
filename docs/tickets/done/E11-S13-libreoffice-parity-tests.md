# E11-S13: LibreOffice headless formula-parity test harness

## Goal
Add a test harness that recalculates the written XLSX/ODS `Allocation` sheet with `soffice --headless`, exports the recalculated values, and diffs them against `allocate()`'s pure-Python output — RFC §8 calls this "the core of the test suite" since openpyxl/odfpy never evaluate formulas themselves.

## Context
- Read: docs/rfc_allocation_sheet.md §8 (parity-testing paragraph — exact masks to test: all-enabled, each single-signal-disabled mask for a sample day, ~100 random masks; float tolerance; "This one test simultaneously validates the XLSX and ODS writers, since LibreOffice reads both")
- Files to modify: tests/unit/test_allocation_parity.py (new); may add a small helper script/module e.g. strategy/allocation_parity_helper.py if the soffice-invocation logic is reusable (implementer's call)
- Depends on: E11-S07 (`allocate` reference), E11-S09 (`write_xlsx_sheet`), E11-S10 (`write_ods_sheet`)
- This test requires `soffice` (LibreOffice) to be installed on the machine running it — check availability at test-collection time (e.g. `shutil.which("soffice")`) and `pytest.skip(...)` the whole module if unavailable, so the rest of the suite isn't blocked on environments without LibreOffice (per CLAUDE.md's "no GPU or model download needed" spirit for `tests/unit/` — this is the one exception needing an external binary, so it must degrade gracefully)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Helper to recalculate: write a workbook to a temp file, invoke `soffice --headless --convert-to csv:"Text - txt - csv (StarCalc)":44,34,0,1,,,,,,,,-1 --outdir <tmp> <file>` (or an equivalent forced-recalculation conversion — implementer should verify empirically that the chosen conversion actually forces recalculation rather than reading cached/stale values, since some conversion filters don't recalc by default; a `soffice --headless --calc --convert-to ...` variant or a macro-based recalc may be needed — this is the trickiest part of the story, budget time for it), read back the resulting values
- Test 1: all-enabled mask — build candidates from a small fixture (reuse the RFC §7 worked-example rows), run `allocate()`, write both XLSX and ODS via E11-S09/S10, recalculate each with LibreOffice, diff every formula-column value against the corresponding `allocate()` row within float tolerance (e.g. `1e-6` relative or `1e-9` absolute, whichever RFC implies — no explicit tolerance given in RFC, so document the chosen value in the test)
- Test 2: each single-signal-disabled mask for the same fixture day (loop over every ticker, disable just that one, recalculate, diff) — one parametrized test
- Test 3: ~100 random enabled/disabled masks over the fixture (seeded RNG for determinism) — diff every mask's recalculated result against `allocate()`
- Any diff beyond tolerance fails that test case with a clear message identifying which cell/ticker/column mismatched
- Both XLSX and ODS are exercised by every mask (per RFC: "validates the XLSX and ODS writers" — do not skip one format even if the other already passed)

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation_parity.py -q` passes on a machine with `soffice` installed, and cleanly skips (not fails) where it's absent
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
