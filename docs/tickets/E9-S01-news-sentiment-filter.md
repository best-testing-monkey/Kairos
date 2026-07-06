# E9-S01: News Sentiment Filter

## Goal
Implement news sentiment filter that validates signal direction against sentiment context.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §6.5
- Files to modify/create: strategy/kairos_sentiment.py (NEW), tests/unit/test_sentiment.py
- Relevant existing code: None yet (new sentiment module scaffold)
- Standards: see docs/tickets/APPENDIX-A-standards.md (degrades gracefully when context key missing)

## Acceptance criteria
- NewsSentimentFilterStrategy filter wrapper
- Reads context["news_sentiment"][symbol] in [-1,1]
- Vetoes signals fighting strong opposing sentiment (|s| > 0.5), verified in `test_news_sentiment_opposing_veto`
- Boosts confidence when aligned with sentiment, verified in `test_news_sentiment_aligned_boost`
- Missing context key -> identical behavior to unwrapped strategy (pass-through), verified in `test_news_sentiment_missing_context_passthrough`
- Both veto and boost paths unit-tested with synthetic context

## Definition of done
- Unit tests in `tests/unit/test_sentiment.py` pass via `uv run --with pytest python -m pytest tests/unit/test_sentiment.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
