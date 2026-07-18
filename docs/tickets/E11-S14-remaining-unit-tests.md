# E11-S14: Golden-file, property, and determinism tests for allocate()

## Goal
Round out `strategy/allocation.py`'s test coverage per RFC §8's last bullet: a golden-file test on the sample rows, a property test that caps are always respected under random masks, and confirmation the determinism/schema tests from earlier stories form a complete suite (fill any gaps, don't duplicate).

## Context
- Read: docs/rfc_allocation_sheet.md §7 (worked example — the golden-file fixture), §8 last bullet ("golden-file on the sample rows, property test that allocations respect all caps under random masks, determinism test, schema validation per SCHEMA_ERROR condition")
- Files to modify: tests/unit/test_allocation.py (append)
- Depends on: E11-S07 (`allocate`) — everything else is already covered by E11-S01 through E11-S07's own story-level tests; this story's job is to (a) add the golden-file fixture test using the exact RFC §7 table as checked-in expected output, and (b) add the property test — check E11-S07's test suite first and only add what's missing, don't duplicate its determinism/schema tests
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- Golden-file test: hard-code the RFC §7 worked-example candidates (NG=F close_direction, NG=F open_gap, REMX, V, CRM, with the exact risk/reward/b/n/shrink/EV-net/Kelly/Score values from that table) as a fixture, run `allocate()`, assert every derived value matches the RFC's table to a documented tolerance (e.g. `1e-2` given the RFC rounds to 2 decimal places in its own table) — if E11-S07 already added this exact test, reference it here instead of duplicating (check test_allocation.py first)
- Property test: generate ~50-100 random `Candidate` lists (seeded RNG, varying n/win_rate/entry/stop/target/tickers/clusters within realistic ranges) and random enabled masks; for every generated case, assert: sum of `alloc` across selected rows `<= config.gross_cap_pct + tolerance`, no single row's `alloc > config.max_pos_pct + tolerance`, no cluster's summed `alloc > config.max_cluster_pct + tolerance`, and `len(selected) <= config.top_k`
- Confirm (do not necessarily re-test) that determinism (same input → same output) and schema-validation coverage already exist from E11-S02/E11-S07 — if a gap is found, add the missing case here rather than assuming it's covered

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- `uv run --with pytest python -m pytest tests/unit/ -q` full suite passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
