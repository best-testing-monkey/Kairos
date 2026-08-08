# E6-S18 — Populate papertrade_signals by unpacking signals_cache

**Goal:** Read existing `signals_cache` rows over a date window and unpack them into individual `papertrade_signals` rows, cache-aware (idempotent re-runs).

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.1.
- Query `data/pipeline_results.db`'s `signals_cache` table schema directly before writing code (`sqlite3.connect('data/pipeline_results.db')`, `PRAGMA table_info(signals_cache)` or `SELECT sql FROM sqlite_master WHERE name='signals_cache'`) — do not trust this ticket's paraphrase of column names as gospel. Columns include `cache_key, strategy_name, assets, interval, as_of, lookback, pred_samples, min_ev_pct, model_label, model_path, checkpoint_fingerprint, stats_json, advice_json, skipped_json, created_at` as of this writing — `stats_json`/`advice_json` are JSON-encoded lists of per-signal dicts, one list entry per candidate in that group/date.
- Read `strategy/allocation.py`'s `fetch_signals(stats_rows, advice_rows)` (~line 222) — this is the reference for which fields matter per individual signal (`strategy`, `symbol`→ticker, `direction`, `entry`, `stop`, `target`, `expected_value`, `base_win_rate`; from advice_rows: `base_signals`/`oracle_signals` fallback for `n`) and the exclusion rule (`direction.upper() == "FLAT"` rows are skipped). This story does the equivalent unpacking, but writes to a table instead of building `Candidate` objects.
- Read `strategy/kairos_signal_replay.py` (this story's own module, output of E6-S17) for `_ensure_signal_replay_tables`.

**Acceptance criteria:**
- [ ] `unpack_signals_cache_to_papertrade_signals(conn, start_date, end_date) -> int` (returns count of NEW rows inserted): queries `signals_cache` for rows with `as_of` in `[start_date, end_date]`, parses `stats_json`/`advice_json` (`json.loads`), and for each non-FLAT-direction `(stats_row, advice_row)` pair (matched by list index, same convention as `fetch_signals`), inserts one `papertrade_signals` row.
- [ ] `signal_id` is a deterministic hash (e.g. `hashlib.sha256` of a stable string built from `strategy_name`, `ticker`, `direction`, `interval`, `as_of`, `entry`, `stop`, `target`, `model_label`, `checkpoint_fingerprint`) so re-running over an already-covered window does not duplicate rows — use `INSERT OR IGNORE` (or check-before-insert, implementer's call, document which) keyed on this `signal_id` as `PRIMARY KEY`.
- [ ] `source_cache_key` column set to the originating `signals_cache.cache_key`.
- [ ] Calling the function twice over the exact same window inserts 0 new rows the second time (returns `0`).
- [ ] Unit tests (`tests/unit/test_kairos_signal_replay.py`) build a synthetic `signals_cache` table (2-3 rows, hand-built `stats_json`/`advice_json`, including at least one stats_row with `direction: "FLAT"` to verify exclusion) in a throwaway sqlite3 connection, call the function, and assert the resulting `papertrade_signals` rows match expectations exactly (count, field values). A second test asserts the idempotent re-run behavior.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Unpack + idempotency tests pass.
- [ ] Changes committed and `docs/todo.md` E6-S18 item checked off.
