# E10-S04 — Turn `interval_probe_ok` into a real gate; write the universe-1h playbook

**Goal:** `run_stage_universe`'s `interval_probe_ok` value is computed but never affects `passed`; make it a real gate, then document the finished E10 (universe-for-1h) flow as a playbook.

**Context:**
- Depends on E10-S02/S03 (native-interval fetch + scaled thresholds) — do those first.
- `strategy/kairos_pipeline.py`, `run_stage_universe` (~line 503-566). The probe block (~line 530-542):
  ```python
  interval_probe_ok = True
  if interval != "1d":
      try:
          probe_start = end_dt - timedelta(days=5)
          probe = price_cache.get_price_data(
              symbol, start_date=probe_start.isoformat(),
              end_date=end_dt.isoformat(), interval=interval,
          )
          interval_probe_ok = probe is not None and not probe.empty
      except Exception:
          interval_probe_ok = False
      row["interval_probe_ok"] = interval_probe_ok

      passed, fail_reason, liquidity_note = evaluate_liquidity(...)
      row["passed"] = passed
      ...
  ```
  Note: after E10-S02, the MAIN fetch already uses `interval=interval`, so this probe is now largely redundant for `interval != "1d"` (if the main fetch failed, `df is None or df.empty` already short-circuits to `fail_reason = "no_data_returned"` earlier in the function, ~line 518-520, and `evaluate_liquidity` is never even called). Keep the probe block (it's cheap, and it's still useful as an EXTRA data-availability check independent of the stats computation) but make its result actually matter.
- `row["interval_probe_ok"]` is already persisted to the `universe_screen` table (`insert_universe_row`, has an `interval_probe_ok` column per the CREATE TABLE at ~line 115-127) — no schema change needed, this story is pure logic.

**Acceptance criteria:**
- [ ] After the probe block runs (only when `interval != "1d"`, matching existing behavior — for `interval == "1d"` there's no probe and `passed` is exactly the `evaluate_liquidity` result as before, byte-identical), if `interval_probe_ok is False`, override: `row["passed"] = False` and set `row["fail_reason"] = "interval_probe_failed"` (only if `evaluate_liquidity` had otherwise passed — don't clobber a more specific existing `fail_reason` from `evaluate_liquidity` itself; only set this when the row would otherwise have `passed=True`).
- [ ] For `interval == "1d"`: no behavior change (probe block doesn't run at all today for `"1d"`, per the `if interval != "1d":` guard — confirm this guard is untouched).
- [ ] Unit test: mock the probe's `price_cache.get_price_data` to return `None`/empty specifically for the probe call (while the main stats fetch succeeds normally) for `interval="1h"`; assert the resulting row has `passed=False` and `fail_reason="interval_probe_failed"` even though liquidity stats alone would have passed.
- [ ] Write `docs/playbooks/hourly-universe-screen.md` (new file, mirror the structure/tone of `docs/playbooks/hourly-signals.md`): prerequisites, the exact command (`uv run ./strategy/kairos_pipeline.py --stage universe --interval 1h`), what a successful run looks like (row counts, `PASS`/`fail` printout format already in `run_stage_universe`'s own print statements), and a caveats section noting: yfinance's 729-day 1h history cap, the daily-equivalent dollar-volume normalization from E10-S03 (so a human reading a `low_dollar_volume` failure understands the `daily_equiv=` figure in the message), and that this is the FIRST stage in the pipeline — `--stage correlation --interval 1h` (E11) depends on this stage's `universe_screen` rows existing first.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py`).
- [ ] New test passes; full suite green.
- [ ] `docs/playbooks/hourly-universe-screen.md` exists and reads clearly to someone who has never run this stage before.
- [ ] Changes committed and `docs/todo.md` E10-S04 item checked off.
