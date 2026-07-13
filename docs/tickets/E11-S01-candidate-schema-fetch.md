# E11-S01: Candidate schema + fetch_signals()

## Goal
Create `strategy/allocation.py` with a `Candidate` dataclass matching RFC §3's required schema, and `fetch_signals(stats_rows, advice_rows) -> list[Candidate]` adapting the existing signals-report row data into it.

## Context
- Read: docs/rfc_allocation_sheet.md §3 (schema table), §3.1 (config table, for reference only — config itself is E11-S03's job)
- Files to modify: strategy/allocation.py (new); tests/unit/test_allocation.py (new)
- Existing data to adapt (read strategy/kairos_signals.py:217-231 `STATS_COLUMNS`/`build_stats_table`, and the `stats_rows.append({...})` / `advice_rows.append({...})` blocks around kairos_signals.py:670-695 for exact keys currently available): strategy=`row["strategy"]`, ticker=`row["symbol"]`, direction=`row["direction"]` (LONG/SHORT/FLAT — map to enum long/short, exclude FLAT rows entirely from `fetch_signals`), entry/stop/target=as-is, ev_pct=`_ev_pct_value(expected_value, entry)` (reuse this exact helper from kairos_signals.py, do not reimplement), base_win_rate=as-is, n=`base_signals` (fallback `oracle_signals` if missing, same fallback order as `build_signals_table` in kairos_signals.py:273-300), backtest_period=as-is, sharpe=`base_sharpe`, advised_liquidity_pct=`size * 100`.
- Fields NOT present in current data (RFC marks nullable in v1): avg_win_pct, avg_loss_pct, avg_holding_days — set to `None` in `Candidate`.
- Standards: docs/tickets/APPENDIX-A-standards.md

## Acceptance criteria
- `Candidate` is a dataclass (not a dict) with exactly the fields in RFC §3's table (rename `ticker` per RFC; keep `direction` as a `str` "long"/"short", not FLAT) plus `avg_win_pct`, `avg_loss_pct`, `avg_holding_days` all `Optional[float] = None`
- `fetch_signals(stats_rows, advice_rows)` returns one `Candidate` per non-FLAT stats_row, correctly zipped to its corresponding advice_row (same pairing order the existing `run()` loop produces them in — i.e. by list index, since both lists are appended in lockstep in kairos_signals.py's `run()`)
- `risk_pct`/`reward_pct` are NOT computed here (that's E11-S03) — this story only builds the `Candidate` schema and fetch adapter
- Unit tests (no GPU/network, no DB): construct small `stats_rows`/`advice_rows` fixtures by hand (mirroring the exact dict shapes from kairos_signals.py's `run()`), assert `fetch_signals` produces correctly-populated `Candidate` objects, FLAT-direction rows are excluded, and `n` falls back to `oracle_signals` when `base_signals` is missing (same as `build_signals_table`'s existing fallback)

## Definition of done
- `uv run --with pytest python -m pytest tests/unit/test_allocation.py -q` passes
- Standards followed, committed per APPENDIX-A, story checked off in docs/todo.md
