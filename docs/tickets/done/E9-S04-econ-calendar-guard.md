# E9-S04: Econ Calendar Guard

## Goal
Implement economic calendar filter managing entry and position risk around high-impact events.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.5
- Files to modify/create: strategy/kairos_sentiment.py, tests/unit/test_sentiment.py
- Relevant existing code: None yet for econ calendar logic
- Standards: see docs/tickets/APPENDIX-A-standards.md (degrades gracefully when context key missing)

## Acceptance criteria
- EconCalendarGuardStrategy filter wrapper
- Reads context["econ_events"] = list of {date, impact}
- Vetoes new entries the day before high-impact events (CPI/NFP/FOMC), verified in `test_econ_calendar_pre_event_veto`
- Tightens stops on open positions via metadata flag (for implementation at execution layer)
- Missing context key -> pass-through behavior, verified in `test_econ_calendar_missing_context_passthrough`
- Empty context["econ_events"] list -> pass-through, verified in `test_econ_calendar_empty_events_passthrough`
- Stop-tightening metadata flag verified on open positions, verified in `test_econ_calendar_stop_tightening_flag`

## Definition of done
- Unit tests in `tests/unit/test_sentiment.py` pass via `uv run --with pytest python -m pytest tests/unit/test_sentiment.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
