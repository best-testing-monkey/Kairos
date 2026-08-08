# E10-S04: CLI wiring for stage auto

## Goal
Wire `--stage auto` and its flags into the kairos_pipeline.py argparse and main dispatch.

## Context
- Read: strategy/DESIGN_DOC_pipeline_automation.md §2.1; strategy/kairos_pipeline.py (existing argparse block and stage dispatch in __main__ / main())
- Files to modify: strategy/kairos_pipeline.py; tests/unit/test_pipeline_auto.py (append)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `--stage auto` accepted; new flags: `--intervals` (nargs="+", default ["1d"]), `--min_sharpe` (float, 0.0), `--min_signals` (int, 3), `--force`, `--skip_universe`, `--report_only` (all store_true defaults False)
- `--stage auto --interval 1h` → argparse error (singular flag rejected with auto); `--intervals` with any non-auto stage → argparse error (`test_cli_flag_exclusivity` — drive via parse_args on an extracted/parametrized parser or subprocess `--help`/error exit codes)
- `--report_only` dispatches to build_viability_report only — assert run_stage_auto NOT called (monkeypatch)
- Dispatch passes all flags through to run_stage_auto verbatim (recording stub)
- Existing single-stage invocations unchanged: `--stage oracle --assets A B` still dispatches identically (regression test with recording stub)
- `uv run ./strategy/kairos_pipeline.py --help` exits 0 and lists the new flags (subprocess test, no GPU)

## Definition of done
- `timeout 120 uv run --with pytest python -m pytest tests/unit/test_pipeline_auto.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
