# E11-S01 — Scale correlation's `min_overlap`/`roll_window` by interval

**Goal:** `compute_pair_correlation`'s `min_overlap=150`/`roll_window=30` defaults are bar counts sized for daily bars (150 bars ≈ 150 trading days); at `1h` the same raw bar counts cover only ~6 calendar days, far too short a window for a meaningful correlation estimate. Scale both by the interval.

**Context:**
- Depends on E10 (universe-for-1h) being done, since correlation reads `universe_screen` survivors.
- `strategy/kairos_pipeline.py`, `compute_pair_correlation(series_a, series_b, min_overlap=150, roll_window=30)` (~line 571) — pure function, called from `run_stage_correlation` (~line 780) with no override today: `full_corr, rolling_median, overlap = compute_pair_correlation(closes[a], closes[b])`.
- `strategy/kairos_pipeline.py`, `run_stage_correlation(conn, asset_class_filter=None, interval="1d", min_abs_corr=None)` (~line 717) already has `interval` in scope at the call site.
- `strategy/kairos_backtest.py`'s `BARS_PER_DAY` dict (already imported into `kairos_pipeline.py` via `kairos_strategies`, per E10-S01) is the scaling factor: for `1d`, `BARS_PER_DAY["1d"] == 1` (no-op); for `1h`, `BARS_PER_DAY["1h"] == 24`.
- Do NOT change `compute_pair_correlation`'s own defaults (150/30) — those stay the "1d-bar-count" meaning for any other/direct caller (e.g. existing unit tests calling it without overrides must keep passing unchanged). Instead, scale at the `run_stage_correlation` call site only.

**Acceptance criteria:**
- [ ] `run_stage_correlation`'s call becomes: `full_corr, rolling_median, overlap = compute_pair_correlation(closes[a], closes[b], min_overlap=int(150 * BARS_PER_DAY.get(interval, 1)), roll_window=int(30 * BARS_PER_DAY.get(interval, 1)))`.
- [ ] For `interval="1d"`: `min_overlap=150`, `roll_window=30` — identical to the current hardcoded defaults, so `1d` correlation output is byte-identical to before this change.
- [ ] For `interval="1h"`: `min_overlap=3600`, `roll_window=720` (i.e. still ~150/~30 calendar days of coverage, matching the `1d` case's real-world window length).
- [ ] Unit test: call `run_stage_correlation` with a mocked `price_cache.get_price_data` returning enough synthetic bars, once with `interval="1d"` and once with `interval="1h"`; assert (via monkeypatching `compute_pair_correlation` to record its kwargs, or inspecting `MIN_ABS_CORR`-independent behavior) that the `min_overlap`/`roll_window` values passed differ by exactly the `24x` `BARS_PER_DAY` ratio between the two calls.
- [ ] Existing `TestCorrelationIntervalThreading`/`TestCorrelationSingletonsAndCross`/`test_correlation_does_not_leak_across_interval_universe_runs` tests in `tests/unit/test_pipeline_auto.py` (added in the E0 slice of this design doc) still pass unmodified.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py`).
- [ ] New + existing correlation tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E11-S01 item checked off.
