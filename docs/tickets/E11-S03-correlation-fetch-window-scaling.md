# E11-S03 — Scale correlation's price-fetch window to match its scaled min_overlap

**Goal:** E11-S02's live verification found that `--stage correlation --interval 1h` currently produces **zero pairs on every run** — E11-S01 scaled `min_overlap`/`roll_window` 24x for 1h, but the price-series fetch window that feeds `compute_pair_correlation` was never scaled to match, so it's structurally impossible for any pair to clear the new `min_overlap` threshold. Fix the fetch window so it scales the same way.

**Context:**
- Read `docs/playbooks/hourly-correlation.md`'s "⚠️ Known bug" section (written by E11-S02, live-verified 2026-08-20) for the full root-cause writeup — read it before touching code, it already did the diagnosis.
- `strategy/kairos_pipeline.py`, `run_stage_correlation` (~line 736-753, just before the `for symbol, ac in survivors:` fetch loop). Current code:
  ```python
  bars_needed = 400
  days_needed = calendar_days_for_bars(bars_needed, bars_per_day, "BTC-USD", buffer_days=0)
  end_dt = date.today()
  start_dt = end_dt - timedelta(days=days_needed)
  ```
  `bars_per_day = BARS_PER_DAY.get(interval, 1)` is already computed just above this (added in an earlier E0 slice of this design doc). For `interval="1d"`, `bars_per_day=1`, so `days_needed ≈ 400` calendar days ≈ 400 daily bars fetched — matches the un-scaled `min_overlap=150` from before E11-S01 with plenty of headroom. For `interval="1h"`, `bars_per_day=24`, so `days_needed ≈ 400/24 ≈ 17` calendar days ≈ only ~400 HOURLY bars fetched — but E11-S01 (commit `b082e53`) now requires `min_overlap=3600` bars of overlap at `1h`. `400 < 3600` always, so `compute_pair_correlation` always returns `(None, None, overlap_bars)`.
- The fix: `bars_needed` itself needs to scale by the same factor E11-S01 used for `min_overlap`/`roll_window`, so the fetch window and the overlap requirement stay consistent at every interval. E11-S01's scaling factor was `BARS_PER_DAY.get(interval, 1)` applied to the base 1d bar counts (`150`, `30`). Apply the same factor to `bars_needed`'s base value (`400`): `bars_needed = int(400 * BARS_PER_DAY.get(interval, 1))`.
- Sanity check the result: for `interval="1h"`, `bars_needed = 400 * 24 = 9600` bars, and `calendar_days_for_bars(9600, bars_per_day=24, ...)` should work out to roughly the same ~400 calendar days as the `1d` case (since both bars_needed and bars_per_day scaled together) — confirm this arithmetic holds when you implement it; `calendar_days_for_bars` divides `bars_needed / bars_per_day` internally so the `bars_per_day` scaling cancels out and calendar-day coverage stays ~400 days regardless of interval, which is the intended, consistent behavior.
- Also check yfinance's 729-day cap for `1h` (`_YF_MAX_DAYS["1h"] = 729`, same file, ~line 1160) — 400 calendar days is safely under that cap, no additional capping needed here (unlike E10-S02's universe fetch, which explicitly capped at `_YF_MAX_DAYS` because its base window was `400` days already at the boundary; here the *bar count* scales, not the calendar-day window, so the calendar-day window stays ~400 regardless of interval — same as today's `1d` behavior).

**Acceptance criteria:**
- [ ] `bars_needed = int(400 * BARS_PER_DAY.get(interval, 1))` replaces the hardcoded `bars_needed = 400`.
- [ ] For `interval="1d"`: `BARS_PER_DAY["1d"] == 1`, so `bars_needed == 400` — byte-identical to before this change.
- [ ] For `interval="1h"`: `bars_needed == 9600`, and the resulting `days_needed` (via `calendar_days_for_bars`) works out to roughly the same ~400-day calendar window as the `1d` case (not ~17 days as today).
- [ ] Unit test: mock `price_cache.get_price_data` to record the `start_date`/`end_date` window it was called with; call `run_stage_correlation` once with `interval="1d"` and once with `interval="1h"`; assert the calendar-day SPAN of the two windows is approximately equal (within a small tolerance for the FX/crypto 7/5-day padding in `calendar_days_for_bars`), rather than the `1h` window being ~24x narrower.
- [ ] Live re-verification (you can do this yourself as part of this story, since it's fast and non-GPU): re-run `uv run ./strategy/kairos_pipeline.py --stage correlation --interval 1h` against whatever `1h` universe survivors currently exist in `data/pipeline_results.db`, and confirm `compute_pair_correlation` is no longer structurally guaranteed to return `None` for every pair (i.e. `overlap_bars` in a debug print, or the final pair count, should reflect the wider fetch window). If there are currently very few `1h` universe survivors (per E11-S02's finding that most crypto symbols are failing universe screening due to a separate DST bug), a handful of survivors is enough to prove the fix — you don't need a large survivor set, just confirm the mechanism works.
- [ ] Update `docs/playbooks/hourly-correlation.md`: remove or strike through the "⚠️ Known bug" section (since it's now fixed) and replace it with a note that it was found 2026-08-20 and fixed by this story, plus fresh output from your live re-verification run.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_pipeline.py`).
- [ ] New + existing tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] `docs/playbooks/hourly-correlation.md` updated to reflect the fix.
- [ ] Changes committed and `docs/todo.md` E11-S03 item checked off.
