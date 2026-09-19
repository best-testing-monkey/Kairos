# E18-S01 — Grid-candidate hindsight labels for historical signals

**Goal:** For each already-precomputed historical signal (from the offline-replay tables), resolve
a small grid of alternative `(stop_pct, target_pct)` candidates against that signal's real
subsequent price path, and persist a binary "did target hit before stop" label per
(signal, candidate) — the training labels E18-S03 will consume. Price-path-only; no model/GPU
access needed.

**Context:**
- Read `docs/tickets/DESIGN_DOC_ml_tpsl_optimization.md` §4 ("Label" paragraph, revised
  2026-09-19) for the intent.
- `strategy/kairos_signal_replay.py` is the tool to mirror, not call as-is:
  - `compute_closure(conn, signal_row, interval_ladder, fee_pct, slippage_pct, db_path,
    engine_version)` (~line 443-613) resolves ONE signal's OWN stored stop/target against forward
    bars and writes to `papertrade_signals_closure`. Read its body fully — it's the bar-walk
    pattern to reuse, via `BacktestEngine(predictor=None, fee_pct=..., slippage_pct=...)` and its
    private `_check_exit`/`_calculate_pnl` (`kairos_backtest.py:2061`, `:2091`; class starts
    `kairos_backtest.py:1962`).
  - **Do not write grid-candidate results into `papertrade_signals_closure`** — that table is
    keyed 1:1 on `signal_id` and is consumed by the real replay loop (`kairos_signal_replay.py`'s
    `replay()`); mixing synthetic per-candidate rows into it would corrupt that consumer. Either
    extract a shared resolution helper from `compute_closure`'s body, or duplicate its bar-walk
    loop in this story's own module — implementer's call, whichever is the smaller diff.
  - `_ensure_configured_db(db_path)` (~line 48) — call this before any `price_cache` lookup, same
    as every other caller in this codebase (`price_cache.configure(remote=False, ...)`).
  - `resolve_interval_for_signal(...)` (~line 287) — the per-signal interval-ladder fallback; reuse
    it rather than assuming a fixed interval.
- Source data: the `papertrade_signals` table (schema in
  `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.1) — columns include `signal_id`,
  `ticker`, `direction`, `interval`, `as_of`, `entry`, `stop`, `target`, `strategy_name`,
  `model_label`. Assume `--precompute` has already been run and this table is populated; do not
  re-derive it from `signals_cache` in this story.
- New table, `pipeline_results.db`:
  ```sql
  CREATE TABLE IF NOT EXISTS tpsl_label_candidates (
      signal_id       TEXT NOT NULL,
      stop_pct        REAL NOT NULL,
      target_pct      REAL NOT NULL,
      resolved        INTEGER NOT NULL,   -- 0 if disqualified (no interval resolved, or ran out of bars)
      hit_target_first INTEGER,           -- 1/0, NULL when resolved=0
      interval_used   TEXT,
      engine_version  TEXT NOT NULL,
      computed_at     TEXT NOT NULL,
      PRIMARY KEY (signal_id, stop_pct, target_pct)
  );
  ```
- Candidate grid (hardcode as a module constant, keep small for v1):
  `STOP_PCT_GRID = [10.0, 15.0, 20.0]`, `TARGET_PCT_GRID = [75.0, 85.0, 90.0]` (9 combinations per
  signal) — matches the existing `pct_X` convention (`kairos_backtest.py`'s `_compute_stats()`
  percentile keys).
- For each candidate: derive the candidate's stop/target **prices** from the signal's own `entry`
  and `direction` using the same percentile-of-entry convention strategies already use (e.g. for a
  LONG, `candidate_stop = entry * (1 - stop_pct/100)`, `candidate_target = entry * (1 +
  target_pct/100)`; mirror direction-reversal for SHORT the way `kairos_sentiment.py:284-288`
  does it) — then resolve exactly like `compute_closure` does, but against these candidate prices
  instead of the signal's stored `stop`/`target`.
- New script location: `scripts/tpsl_label_grid.py` (mirrors `scripts/ab_exit_rule.py`,
  `scripts/backfill_class_stats.py` — standalone `pipeline_results.db` analysis tools already live
  in `scripts/`).

**Acceptance criteria:**
- [ ] `tpsl_label_candidates` table created (idempotent `CREATE TABLE IF NOT EXISTS`).
- [ ] Script iterates every `papertrade_signals` row not yet fully covered in
  `tpsl_label_candidates` for the current `engine_version`, computes all 9 grid candidates per
  signal, and writes one row per (signal, candidate) with `INSERT OR REPLACE` (idempotent re-run).
- [ ] `resolved=0` candidates (interval ladder exhausted, or price data runs out before either
  barrier triggers) get `hit_target_first=NULL`, matching `papertrade_signals_closure`'s existing
  disqualification convention (`compute_closure`'s docstring) — no fabricated label.
- [ ] Unit test: synthetic price path with a known outcome for at least 2 candidates (one where
  target is hit first, one where stop is hit first) — `hit_target_first` matches hand-derivation.
- [ ] Unit test: a candidate whose bracket never resolves within the available synthetic bars gets
  `resolved=0`, `hit_target_first=NULL` — not a fabricated 0/1.
- [ ] Unit test: re-running the script over the same window with the same `engine_version` writes
  zero new/changed rows (idempotency, same discipline as `kairos_signal_replay.py`'s own
  `--precompute` cache-reuse test).
- [ ] Uses `_ensure_configured_db()`/`price_cache.configure(remote=False, ...)` before any price
  lookup — a missed call here silently disqualifies every candidate (see this repo's CLAUDE.md
  "Like every other price_cache caller..." note on `kairos_signal_replay.py`'s own past bug of
  exactly this shape).

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `scripts/tpsl_label_grid.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Changes committed and `docs/todo.md` E18-S01 item checked off.
