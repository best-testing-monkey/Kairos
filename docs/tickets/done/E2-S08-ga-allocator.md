# E2-S08: GA Allocator

## Goal
Implement genetic algorithm-based portfolio allocator with weekly caching and weekly refitting.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §2.8
- Files to modify/create: strategy/kairos_portfolio.py, tests/unit/test_portfolio.py
- Relevant existing code: PortfolioAllocator base class from E1-S02
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- GAAllocator class with fitness = trailing 60-day Sharpe of weight vector applied to returns
- Population 50, tournament selection, blend crossover, Gaussian mutation σ=0.05, 20 generations
- Refitted weekly, cached otherwise (deterministic seed derived from date), verified in `test_ga_weekly_cache`
- Fitness non-decreasing across generations on fixed data, verified in `test_ga_fitness_monotonic`
- Weekly cache verified: identical output within the week, verified in `test_ga_same_date_same_output`
- Respects §2.1-style caps (gross_cap, max_weight), verified in `test_ga_respects_caps`

## Definition of done
- Unit tests in `tests/unit/test_portfolio.py` pass via `uv run --with pytest python -m pytest tests/unit/test_portfolio.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
