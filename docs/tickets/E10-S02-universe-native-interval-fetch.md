# E10-S02 — `run_stage_universe` fetches native-interval bars instead of hardcoded 1d

**Goal:** `run_stage_universe` in `strategy/kairos_pipeline.py` always fetches `interval="1d"` for its liquidity/volatility computation regardless of the `--interval` argument; make it fetch and compute from the actual requested interval.

**Context:**
- Depends on E10-S01 (`compute_universe_stats` gains an `interval` parameter) — do that story first, or verify it's already merged before starting this one.
- `strategy/kairos_pipeline.py`, function `run_stage_universe(conn, interval="1d")` (search `def run_stage_universe`, ~line 503). Current fetch call (~line 514-517):
  ```python
  df = price_cache.get_price_data(
      symbol, start_date=start_dt.isoformat(), end_date=end_dt.isoformat(),
      interval="1d",
  )
  ```
  This is hardcoded to `"1d"` even though the function already receives `interval` as a parameter and uses it elsewhere (e.g. `start_run(conn, "universe", interval, ...)`, and the separate `interval_probe_ok` probe a few lines below that already fetches with the real `interval`).
- There is a SEPARATE probe block right after (search `interval_probe_ok`, ~line 530-542) that fetches 5 days of the real `--interval` data purely to check "can we fetch this interval at all" — this is now redundant once the main fetch itself uses the real interval; that probe block's cleanup is E10-S04's job, not this one. Leave it in place for this story (don't remove it), just make the MAIN fetch (the one whose output feeds `compute_universe_stats`) use `interval` instead of the hardcoded `"1d"`.
- `end_dt - timedelta(days=400)` (the `start_dt` computation, ~line 507) is a calendar-day lookback window sized for daily bars (400 days ≈ 400 daily bars). For `1h`, 400 calendar days would try to fetch ~9600 hourly bars, which exceeds yfinance's 729-day cap for 1h history (`kairos_pipeline.py`'s own `_YF_MAX_DAYS` dict, ~line 1160, already has `"1h": 729`). Cap `start_dt`'s lookback at `min(400, _YF_MAX_DAYS.get(interval, 400))` days so a 1h universe run doesn't request more history than the data source can return. Import `_YF_MAX_DAYS` is already module-local (same file), no new import needed.

**Acceptance criteria:**
- [ ] Main fetch call passes `interval=interval` (the function's own parameter) instead of the hardcoded `interval="1d"`.
- [ ] `compute_universe_stats(df)` call becomes `compute_universe_stats(df, interval=interval)`.
- [ ] `start_dt` lookback window is `end_dt - timedelta(days=min(400, _YF_MAX_DAYS.get(interval, 400)))`.
- [ ] Calling `run_stage_universe(conn, interval="1d")` (the existing default/only-used-today path) produces byte-identical DB rows to before this change — verify by running the existing `tests/unit/test_pipeline_auto.py` universe-stage tests unchanged and confirming they still pass without modification.
- [ ] New unit test: mock `price_cache.get_price_data` to record every `interval=` kwarg it was called with; call `run_stage_universe(conn, interval="1h")` against a `temp_db` fixture with a couple of `CANDIDATE_UNIVERSE` symbols; assert the MAIN stats fetch call (not the probe call) used `interval="1h"`.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py`).
- [ ] All `tests/unit/test_pipeline_auto.py` universe-stage tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E10-S02 item checked off.
