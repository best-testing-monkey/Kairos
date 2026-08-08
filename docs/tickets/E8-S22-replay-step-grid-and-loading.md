# E8-S22 — Data-driven replay step grid & per-step candidate loading

**Goal:** Derive the replay loop's step sequence from the ACTUAL `as_of` timestamps present in the data (not a synthesized daily calendar), and load one step's tradeable candidates.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.4 IN FULL — especially the `replay_steps = sorted(DISTINCT as_of FROM papertrade_signals WHERE interval = :interval ...)` sketch and its explicit warning against `range(start_ts, end_ts, timedelta(days=1))`-style fixed-cadence assumptions. A single replay run is scoped to ONE `interval` value (comparing cadences means two separate replay invocations, not one that blends both).
- Read `strategy/kairos_signal_replay.py` (own module — schema from E6-S17, populated by E6-S18/E7-S21).
- Read `strategy/allocation.py`'s `fetch_signals(stats_rows, advice_rows)` (~line 222) to see the EXACT `stats_row`/`advice_row` dict shape it expects (keys: `strategy`, `symbol`, `direction`, `entry`, `stop`, `target`, `expected_value`, `base_win_rate` for stats_rows; `expected_value`, `entry`, `base_win_rate`, `base_signals`, `oracle_signals`, `signal` for advice_rows) — this story's `load_step_candidates` must reconstruct dicts in this exact shape from `papertrade_signals`/`papertrade_signals_closure` columns so E8-S23 can feed them into `fetch_signals` UNCHANGED.

**Acceptance criteria:**
- [ ] `replay_steps(conn, interval: str, start_ts, end_ts) -> list[str]`: returns the SORTED, DISTINCT `as_of` values present in `papertrade_signals` for the given `interval`, within `[start_ts, end_ts]`. Does NOT synthesize a fixed-cadence date range — only timestamps that actually have signal data appear.
- [ ] `load_step_candidates(conn, interval: str, as_of: str) -> list[tuple[dict, dict]]`: returns `(stats_row, advice_row)` dict pairs for every `papertrade_signals` row at that `(interval, as_of)` that has a corresponding `papertrade_signals_closure` row with `resolved=1`. Rows with no closure row, or with `resolved=0`, are EXCLUDED entirely — not included with null/zero stats (per DESIGN_DOC §3.2's disqualification rule). The returned dicts' keys match `fetch_signals`'s expected shape exactly (verify against current `fetch_signals` source, not this ticket's paraphrase).
- [ ] Unit tests build a synthetic `papertrade_signals`/`papertrade_signals_closure` fixture in a throwaway sqlite3 connection: (a) a mix of resolved and unresolved signals at the same `as_of` — assert `load_step_candidates` returns only the resolved ones; (b) `as_of` timestamps that are NOT daily-spaced (e.g. `interval="1h"` with timestamps a few hours apart, not calendar-day-spaced) — assert `replay_steps` returns exactly those timestamps, proving the step grid is genuinely data-driven.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Step-grid and candidate-loading tests pass, including the non-daily-spaced case.
- [ ] Changes committed and `docs/todo.md` E8-S22 item checked off.
