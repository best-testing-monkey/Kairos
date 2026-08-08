# E6-S17 — Create papertrade_signals / papertrade_signals_closure schema

**Goal:** Create the `papertrade_signals` and `papertrade_signals_closure` tables in `pipeline_results.db`, via a new module `strategy/kairos_signal_replay.py`.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.1 for the exact schema (both `CREATE TABLE` statements and the index).
- Read `strategy/kairos_papertrade.py`'s `_ensure_mtm_daily_table`/`_insert_mtm_daily_row` (search for `_ensure_mtm_daily_table`, ~line 1030) for the exact direct-`sqlite3`-connection pattern to follow: `conn.execute("CREATE TABLE IF NOT EXISTS ...")`, `conn.commit()`, no ORM.
- Read `strategy/kairos_signals.py`'s `DB_PATH` constant (`os.path.join(REPO_ROOT, "data", "pipeline_results.db")`) — this is where both new tables live, same DB as `signals_cache`/`viability_report`.
- Create new file `strategy/kairos_signal_replay.py`.

**Acceptance criteria:**
- [ ] New module `strategy/kairos_signal_replay.py` created with a module docstring stating its purpose (fast offline replay of selection/allocation rules — no GPU, no live `phantom`).
- [ ] `_ensure_signal_replay_tables(conn) -> None` creates both tables (`CREATE TABLE IF NOT EXISTS`) with the EXACT schema from DESIGN_DOC §3.1: `papertrade_signals` (signal_id, strategy_name, ticker, direction, interval, as_of, entry, stop, target, expected_value, base_win_rate, n, model_label, checkpoint_fingerprint, source_cache_key, created_at) and `papertrade_signals_closure` (signal_id, resolved, interval_used, pct_profit, max_drawdown_pct, trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version), plus `idx_papertrade_signals_as_of`.
- [ ] Function is idempotent — safe to call multiple times without error.
- [ ] Unit test in `tests/unit/test_kairos_signal_replay.py`: use a throwaway `sqlite3.connect(":memory:")` (or `tmp_path`) connection, call the function, verify both tables and the index exist (query `sqlite_master`), and verify a second call doesn't raise.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_signal_replay.py` per APPENDIX-A).
- [ ] Schema unit tests pass.
- [ ] Changes committed and `docs/todo.md` E6-S17 item checked off.
