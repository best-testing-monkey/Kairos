# E2-S12: Orchestrator Allocator Integration

## Goal
Wire portfolio allocators into the strategy orchestrator with proper context passing and disabled-strategy fallback support.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §7.3
- Files to modify/create: strategy/kairos_orchestrator.py (StrategyRegistry at ~line 270), strategy/kairos_portfolio.py, tests/unit/test_orchestrator_allocator.py
- Relevant existing code: StrategyRegistry from commit f0662fd with per-class disabled-strategy fallback, per-asset signals from meta-filter chain
- Standards: see docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- StrategyRegistry gains a register_allocator() method for wiring allocators
- Allocator applied after per-asset signal generation and meta-filters, replacing per-signal size with allocator weight
- Per-signal size becomes within-asset cap, verified in `test_allocator_replaces_per_signal_size`
- New context keys: context["returns_window"] (trailing return panel, computed once per day for all portfolio/econometric/factor strategies)
- New context keys: context["realized_vol"] populated once per day
- Disabled-strategy fallback still works with allocator-driven sizing, verified in `test_allocator_with_disabled_strategy_fallback`
- Allocator applied after meta-filters, not before, verified in `test_allocator_after_meta_filters`

## Definition of done
- Unit tests in `tests/unit/test_orchestrator_allocator.py` pass via `uv run --with pytest python -m pytest tests/unit/test_orchestrator_allocator.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
