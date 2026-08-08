# E8-S23 — Replay loop core (unleveraged)

**Goal:** The actual offline replay: apply a candidate `AllocationConfig`/selection rule against precomputed signal closures, stepping through the data-driven replay grid, tracking a simple cash ledger.

**Context — this phase is UNLEVERAGED ONLY, read the scope note in §1 of the design doc before writing anything:**
- `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §1's "Phase scope: unleveraged only" note and §3.4 IN FULL, especially the "Known simplification" paragraph. `max_leverage` stays at `AllocationConfig`'s default (`1.0`) throughout — this story must NEVER set `max_leverage`/`margin_utilization_cap` to anything else, and must NOT implement any margin-lock, `admission_check`, or liquidation logic. Cash bookkeeping is simple spot/full-notional only.
- Read `strategy/allocation.py`'s `fetch_signals`, `allocate(candidates: list[Candidate], config: AllocationConfig, enabled_mask: dict = None) -> AllocationResult`, and `AllocationConfig`'s current field list (confirm field names/defaults against source — do not assume this ticket's list is exhaustive or current) before writing code.
- Read `strategy/kairos_papertrade.py`'s `compute_final_metrics` (~line 1520) and `build_closed_trade_equity_curve` (~line 1310) for the metrics-dict shape and drawdown-computation CONVENTION this story's output should be comparable to on the SAME axes: `total_profit_eur`, `pct_profit`, `num_trades`, `pct_max_drawdown` at minimum. This story does NOT need any `mtm_*` keys — those are margin/leverage-specific (out of scope, per the phase-scope note).
- Read `strategy/kairos_signal_replay.py` (own module) for `replay_steps`/`load_step_candidates` (E8-S22).

**Acceptance criteria:**
- [ ] `replay(conn, interval: str, start_ts, end_ts, alloc_config: AllocationConfig, starting_capital: float) -> dict`:
  - Iterates `replay_steps(conn, interval, start_ts, end_ts)` in order.
  - At each step: loads candidates via `load_step_candidates`, converts to `Candidate` objects via `fetch_signals(stats_rows, advice_rows)` UNCHANGED, calls `allocate(candidates, alloc_config)` UNCHANGED to get this step's selected, sized rows.
  - For each `SELECTED` row: sizes the position against CURRENT running capital (using `allocate()`'s own sizing output — do not re-derive sizing logic), and applies that signal's precomputed `pct_profit` (from `papertrade_signals_closure`) to the running cash ledger. Simple bookkeeping: `cash_delta = notional * pct_profit / 100.0` (or equivalent — confirm `pct_profit`'s exact units/sign convention against E7-S21's output before assuming a formula).
  - Tracks a running equity curve (one point per replay step).
- [ ] `alloc_config.max_leverage` is asserted or documented to remain `1.0` throughout this function — this is not merely "the default," it is a hard invariant of this story; do not add any code path that would let it be anything else.
- [ ] Computes `pct_max_drawdown` from the tracked equity curve using the SAME peak-to-trough formula convention already used elsewhere in this codebase (check `phantom.reports.metrics.calculate_metrics` or `kairos_papertrade.py`'s own drawdown computation — reuse the existing convention rather than inventing a new one).
- [ ] Returns a dict with at minimum `total_profit_eur`, `pct_profit`, `num_trades`, `pct_max_drawdown`.
- [ ] Unit tests: synthetic `papertrade_signals`/`papertrade_signals_closure` fixture with a handful of signals across 2-3 replay steps and a known `AllocationConfig`; HAND-DERIVE the expected `total_profit_eur`/`num_trades`/`pct_max_drawdown` and assert exact match (`pytest.approx`) — same discipline as this session's `test_leverage_off_matches_pinned_baseline`. Include a case with more candidates than `alloc_config.top_k`/over `max_pos_pct` proving the replay loop's sizing actually RESPECTS `allocate()`'s caps rather than bypassing them (i.e. don't just trust `allocate()` — verify the replay loop uses its output correctly).

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Replay-loop tests pass, including the hand-derived numeric check and the caps-respected case.
- [ ] Changes committed and `docs/todo.md` E8-S23 item checked off.
