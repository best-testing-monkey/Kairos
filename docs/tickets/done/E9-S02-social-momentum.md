# E9-S02: Social Momentum

## Goal
Implement standalone strategy trading social media momentum and detecting blow-off tops.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.5
- Files to modify/create: strategy/kairos_sentiment.py, tests/unit/test_sentiment.py
- Relevant existing code: None yet for social media trading logic
- Standards: see docs/tickets/APPENDIX-A-standards.md (degrades gracefully when context key missing)

## Acceptance criteria
- SocialMomentumStrategy standalone strategy
- Reads context["social_mentions"][symbol] = {count, z_score, sentiment}
- Mention z>3 with positive sentiment -> momentum long (crowd inflow), verified in `test_social_momentum_long`
- z>3 with price already +20%/5d -> fade (blow-off proxy), verified in `test_social_momentum_blowoff_fade`
- Gated on Kronos agreement
- Missing context key -> no signal (returns None), never raises exception, verified in `test_social_momentum_missing_context_no_error`
- Both momentum-long and fade paths exercised in tests

## Definition of done
- Unit tests in `tests/unit/test_sentiment.py` pass via `uv run --with pytest python -m pytest tests/unit/test_sentiment.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
