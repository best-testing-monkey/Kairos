# E10-S06: PIPELINE.md documentation for stage auto

## Goal
Document the auto stage, its flags, resumability semantics, and the viability report in strategy/PIPELINE.md.

## Context
- Read: strategy/DESIGN_DOC_pipeline_automation.md §2.1, §3.2, §3.3, §4; strategy/PIPELINE.md (match its existing structure/tone)
- Files to modify: strategy/PIPELINE.md ONLY
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- New "Stage auto" section covering: the one-command flow, every flag from §2.1 with defaults, resumability/--force semantics, --skip_universe/--report_only, report CSV path pattern and viability_report table, the viability rule (both-sides Sharpe > min_sharpe, min signals), and the note that the report covers ENABLED strategies only (disabled count visible in the built/disabled/evaluating line)
- Worked example command: `uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1d --backtest_period 3m --asset_class crypto`
- Note that stage 5 (finetuned) remains manual and how a finetuned column could later extend the report
- All flag names/defaults verified against the implemented argparse (read strategy/kairos_pipeline.py argparse block; do not copy from the design doc blindly)

## Definition of done
- Doc-only change; `timeout 120 uv run --with pytest python -m pytest tests/unit/test_pipeline_auto.py -q` still passes (sanity)
- Committed per APPENDIX-A, story checked off in docs/todo.md
