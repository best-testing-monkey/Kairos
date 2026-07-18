# E11-S03: AllocationConfig + per-row derived columns

## Goal
Add `AllocationConfig` dataclass (RFC §3.1 defaults) and `compute_derived(c: Candidate, config: AllocationConfig) -> dict` implementing RFC §4.2's per-row formulas to `strategy/allocation.py`.

## Context
- Read: docs/rfc_allocation_sheet.md §3.1 (config table — exact defaults), §4.2 (per-row derived columns — exact formulas, copy them verbatim)
- Files to modify: strategy/allocation.py; tests/unit/test_allocation.py (append)
- Depends on: E11-S01 (`Candidate`)
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `AllocationConfig` dataclass fields and defaults exactly match RFC §3.1's table: `n0=100`, `min_n=50`, `round_trip_cost_pct=0.15`, `kelly_mult=0.35`, `top_k=12`, `max_pos_pct=15`, `max_cluster_pct=25`, `gross_cap_pct=100`, `dust_min_pct=1.0`, `equity=None`, `cluster_map` (dict, default `{}`)
- `compute_derived` returns a dict with keys: `risk_pct`, `reward_pct`, `b`, `loss_pct`, `shrink`, `ev_shrunk`, `ev_net`, `p_shrunk`, `kelly_raw`, `kelly_frac`, `score` — computed exactly per RFC §4.2's pseudocode (including the `avg_win_pct`/`avg_loss_pct`-present branch vs. the geometry-fallback branch for `b`/`loss_pct`)
- Test the geometry-fallback path (avg_win/avg_loss both `None`) and the empirical path (both populated) separately, with hand-computed expected values for at least one candidate each
- Test `shrink`/`ev_shrunk`/`kelly_frac` at boundary `n=0` (shrink should be 0) and large `n` (shrink approaches 1) — no division-by-zero

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
