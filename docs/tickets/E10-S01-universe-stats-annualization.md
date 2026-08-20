# E10-S01 — Make `compute_universe_stats`'s ann_vol annualization interval-aware

**Goal:** `compute_universe_stats` in `strategy/kairos_pipeline.py` hardcodes `np.sqrt(252)` when annualizing volatility, which is only correct for daily bars; make it use the shared `bars_per_year(interval)` helper instead.

**Context:**
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §1–2 (E1/E10 section) for why this matters.
- `strategy/kairos_pipeline.py`, function `compute_universe_stats(df: pd.DataFrame)` (search for `def compute_universe_stats`, ~line 476). Current body:
  ```python
  def compute_universe_stats(df: pd.DataFrame):
      """Compute bars, dollar_volume, ann_vol, atr_pct from a raw OHLCV frame."""
      bars = len(df)
      close = df["close"].astype(float)
      if "volume" in df.columns:
          dollar_volume = float((close * df["volume"].astype(float)).median())
      else:
          dollar_volume = None
      log_ret = np.log(close / close.shift(1)).dropna()
      ann_vol = float(log_ret.std() * np.sqrt(252)) if len(log_ret) > 1 else None
      ...
  ```
- `strategy/kairos_backtest.py` already has `bars_per_year(interval: str) -> float` (returns `BARS_PER_DAY[interval] * 252`; `bars_per_year("1d") == 252` exactly, so this is a safe drop-in replacement for the daily case) — added in the E0 slice of this same design doc. `kairos_pipeline.py` already imports `BARS_PER_DAY` from `kairos_strategies` (which re-exports it from `kairos_backtest`); add `bars_per_year` to that same import line: `from kairos_strategies import asset_class_for, _period_to_weeks, _parse_period, BARS_PER_DAY` → add `bars_per_year`.
- The only caller of `compute_universe_stats` is `run_stage_universe` (search `compute_universe_stats(df)` in the same file, ~line 527) — it does NOT currently pass an interval. This story only changes `compute_universe_stats`'s signature and internals; wiring the real interval through from `run_stage_universe` is E10-S02's job, not this one. To keep this story self-contained and not break the only caller, give the new parameter a default of `"1d"`.

**Acceptance criteria:**
- [ ] `compute_universe_stats(df: pd.DataFrame, interval: str = "1d")` — new optional parameter, default `"1d"`.
- [ ] `ann_vol = float(log_ret.std() * np.sqrt(bars_per_year(interval))) if len(log_ret) > 1 else None` replaces the hardcoded `np.sqrt(252)`.
- [ ] Existing call site (`run_stage_universe`) still compiles unchanged (default parameter means no caller update needed in this story).
- [ ] Unit test in `tests/unit/test_pipeline_auto.py` (or a new `TestComputeUniverseStatsAnnualization` class if none fits): build a small synthetic OHLCV DataFrame, call `compute_universe_stats(df)` (default `interval="1d"`) and assert `ann_vol` is unchanged from calling `compute_universe_stats(df, interval="1d")` explicitly; then call with `interval="1h"` and assert the returned `ann_vol` differs from the `"1d"` result by the expected `sqrt(24)` ratio (`bars_per_year("1h") / bars_per_year("1d") == 24`).

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py` per APPENDIX-A).
- [ ] New/existing tests in `tests/unit/test_pipeline_auto.py` pass, full `tests/unit/` suite still green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Changes committed and `docs/todo.md` E10-S01 item checked off.
