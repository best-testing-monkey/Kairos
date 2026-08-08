# E9-S03: Institutional 13F

## Goal
Implement filter preventing shorts against institutional accumulation.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.5
- Files to modify/create: strategy/kairos_sentiment.py, tests/unit/test_sentiment.py
- Relevant existing code: InsiderCluster and DarkPoolFilter pattern (mentioned as complements)
- Standards: see docs/tickets/APPENDIX-A-standards.md (degrades gracefully when context key missing)

## Acceptance criteria
- Institutional13FFilterStrategy filter wrapper
- Reads context["inst_ownership_delta"][symbol] in quarterly Δ%
- Vetoes shorts against strong institutional accumulation (Δ > +2%), verified in `test_13f_veto_short_vs_accumulation`
- Passes shorts or other signals when Δ below threshold, verified in `test_13f_below_threshold_pass`
- Complements InsiderCluster and DarkPoolFilter (different information source)
- Missing context key -> pass-through behavior, verified in `test_13f_missing_context_passthrough`

## Definition of done
- Unit tests in `tests/unit/test_sentiment.py` pass via `uv run --with pytest python -m pytest tests/unit/test_sentiment.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
