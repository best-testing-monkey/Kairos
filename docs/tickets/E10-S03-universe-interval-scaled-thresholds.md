# E10-S03 — Interval-scale `evaluate_liquidity`'s dollar-volume threshold and min_bars

**Goal:** `evaluate_liquidity`'s dollar-volume threshold is explicitly documented as a **daily** figure, and its `min_bars=200` default means "200 daily bars" (~10 months) today but would mean just ~8 days at `1h` — both need to scale with interval so `1h` universe screening applies an equivalent bar of rigor to `1d`.

**Context:**
- Depends on E10-S02 (real interval flows into `run_stage_universe`'s fetch/stats pipeline) — do that story first.
- `strategy/kairos_pipeline.py`, `liquidity_threshold(asset_class)` (~line 439): docstring literally says "Minimum median **daily** dollar volume, by asset class." Returns `10_000_000.0` for crypto, `50_000_000.0` for equity, `0.0` for fx/other.
- `strategy/kairos_pipeline.py`, `evaluate_liquidity(symbol, asset_class, bars, dollar_volume, ann_vol, atr_pct, min_bars=200, atr_min=0.5)` (~line 448) — pure function, no I/O, already unit-tested with synthetic inputs (good — keep it that way, don't add I/O).
- `compute_universe_stats` computes `dollar_volume` as the **median of `close*volume` per bar** (not per day) — so at `1h`, this is roughly a per-hour dollar volume, ~1/24th the daily-equivalent figure a threshold calibrated for `1d` expects. Similarly `bars` is a raw bar count from the fetched DataFrame, so `min_bars=200` bars means a much shorter real-world lookback at `1h` than at `1d`.
- `strategy/kairos_backtest.py`'s `BARS_PER_DAY` dict (imported already per E10-S01/S02, `BARS_PER_DAY["1h"] == 24`) is the scaling factor to use for both adjustments — this file already imports `BARS_PER_DAY` from `kairos_strategies` (which re-exports it from `kairos_backtest`).
- `evaluate_liquidity` is called from `run_stage_universe` (~line 544) with no `min_bars`/`atr_min` override today — this story adds an `interval` parameter to `evaluate_liquidity` itself and updates that one call site to pass it through, plus normalizes `dollar_volume` before the comparison rather than changing the threshold table (keeps `liquidity_threshold`'s per-asset-class numbers meaningful as "daily" figures, single source of truth).

**Acceptance criteria:**
- [ ] `evaluate_liquidity(..., interval: str = "1d", min_bars: int = 200, atr_min: float = 0.5)` — new `interval` parameter.
- [ ] Before comparing `dollar_volume` against `liquidity_threshold(asset_class)`, normalize it to a daily-equivalent figure: `dollar_volume_daily_equiv = dollar_volume * BARS_PER_DAY.get(interval, 1) if dollar_volume is not None else None` (for `"1d"`, `BARS_PER_DAY["1d"] == 1`, so this is a no-op — behavior unchanged for the existing daily path). Compare `dollar_volume_daily_equiv` against the threshold, but keep the raw `dollar_volume` in the returned `fail_reason` string for debuggability (don't silently substitute the scaled number where a human reads the failure message without context — include both raw and scaled values in the message, e.g. `f"low_dollar_volume(raw={dollar_volume}, daily_equiv={dollar_volume_daily_equiv:.0f}<{threshold})"`).
- [ ] `min_bars` scaling: `run_stage_universe`'s call site passes `min_bars=int(200 * BARS_PER_DAY.get(interval, 1))` explicitly (so `evaluate_liquidity`'s own default of `200` stays the literal daily-bar-count default for any other/direct caller, e.g. existing unit tests that call it without `min_bars`).
- [ ] `run_stage_universe`'s call to `evaluate_liquidity` passes `interval=interval` too.
- [ ] For `interval="1d"`: `dollar_volume_daily_equiv == dollar_volume` and `min_bars` passed from `run_stage_universe` is still `200` — fully unchanged pass/fail behavior.
- [ ] Unit test: synthetic `dollar_volume` that would FAIL the crypto threshold at `1h`'s raw per-bar value but PASS once scaled to daily-equivalent (e.g. `dollar_volume=500_000` at `1h` scales to `12_000_000` daily-equivalent, above the `10_000_000` crypto threshold) — assert `evaluate_liquidity(..., interval="1h")` passes where a naive unscaled comparison would have failed it.
- [ ] Unit test: `interval="1d"` case produces identical `(passed, fail_reason, liquidity_note)` tuples to before this change, for a handful of existing test fixtures in `tests/unit/test_pipeline_auto.py` (search existing `evaluate_liquidity` tests and confirm they still pass unmodified).

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py`).
- [ ] New + existing tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E10-S03 item checked off.
