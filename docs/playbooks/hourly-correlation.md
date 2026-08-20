# Hourly correlation grouping

Group `1h`-screened survivors into correlated clusters — the second stage in the
`--interval 1h` pipeline. Requires [hourly-universe-screen.md](hourly-universe-screen.md)
to have run first.

## Prerequisites

- `--stage universe --interval 1h` must have completed and left `PASS` rows in
  `universe_screen` for this run's `run_id` (check: `sqlite3 data/pipeline_results.db
  "SELECT COUNT(*) FROM runs WHERE stage='universe' AND interval='1h'"`).
- No GPU required — pure price-data correlation math.
- Network access for yfinance (this stage re-fetches close-price series for every
  universe survivor).

## Steps

```bash
uv run ./strategy/kairos_pipeline.py --stage correlation --interval 1h
```

## Bug fix: `min_overlap` scaling + fetch-window scaling (found 2026-08-20, fixed by E11-S03)

A live run on 2026-08-20 found that `--stage correlation --interval 1h` produced **zero
pairs on every run**, with every survivor ending up as an isolated singleton group.
Root cause was a scaling mismatch:

- E11-S01 (commit `b082e53`) scaled `compute_pair_correlation`'s `min_overlap` threshold
  by the bar-per-day factor (`150 × BARS_PER_DAY["1h"] = 3600`), but **never scaled the
  price-series fetch window** (`bars_needed = 400`) to match.
- Result: at `1h`, the stage required 3600 overlapping bars but only fetched ~400,
  making it structurally impossible for any pair to clear the threshold.

Fixed by E11-S03: `bars_needed` now scales by the same interval factor (`int(400 *
BARS_PER_DAY.get(interval, 1))`), so the fetch window and the overlap requirement stay
consistent. For `interval="1h"`, `bars_needed = 9600` bars, resulting in ~400 calendar
days of history — matching the `1d` case.

**Live re-verification (2026-08-20, after fix):**
```
Effective min_abs_corr thresholds: crypto=0.75, default=0.6

Stage 2 (correlation) done: 6 pairs, 4 suggested groups. run_id=730.
  group 1 [fx_commodity] [singleton]: CL=F
  group 2 [fx_commodity] [singleton]: NG=F
  group 3 [fx_commodity] [singleton]: SI=F
  group 4 [fx_commodity] [singleton]: ZW=F
```

The 6 pairs all cleared the `min_overlap` threshold (overlap_bars: 4952–6220, vs
min_overlap 3600). The 4 survivors remain isolated because there aren't enough correlated
pairs *within* each asset class to form multi-symbol groups at this interval; this is
real market structure, not a mechanism bug. `--stage correlation --interval 1h` is now
fully functional.

## Separate finding: crypto universe screening broke for 1h on this same run (2026-08-20) — FIXED by BUG-03

Not a correlation-stage bug, but it's the reason the correlation run above only had
4 non-crypto survivors to work with. A fresh `--stage universe --interval 1h` run
(`run_id=728`) failed **every single crypto symbol except one** (`GALA-USD`, which
failed on `low_dollar_volume` instead) with:

```
fetch_error: Cannot infer dst time from 2025-11-02 01:00:00 as there are no repeated times
```

This was a pandas/timezone DST-transition error in kairos/data.py's
`fetch_price_data_local_fallback` function. The error occurred because the universe
stage's 729-day lookback window (yfinance's 1h history cap) reaches back across the
2025-11-02 US DST fall-back date, and pandas' `tz_localize()` by default (`ambiguous="raise"`)
refused to handle the ambiguous repeated hour. **Fixed by BUG-03 (commit [TBD])**:
`tz_localize("America/New_York")` now includes `ambiguous="infer", nonexistent="shift_forward"`
parameters, allowing it to infer the correct offset from the monotonically increasing order
of the data. An older `1h` universe run from 2026-07-05 (`run_id=44`, 124 survivors including
plenty of crypto) predated this — its 729-day window didn't yet reach back across that DST
boundary. Crypto symbols now either pass or fail for legitimate reasons (low ATR%, insufficient bars)
rather than crashing on the DST error.

## What a (hypothetically) successful run looks like

Once the `min_overlap` bug above is fixed, a healthy run should print something like:

```
Effective min_abs_corr thresholds: crypto=0.75, default=0.6

Stage 2 (correlation) done: <N> pairs, <M> suggested groups. run_id=<id>.
  group 1 [crypto] [BTC-USD, ETH-USD]: mean_corr=0.87
  group 2 [fx_commodity] [singleton]: NG=F
  ...
```

`suggested_groups` rows with more than one symbol and a non-NULL `mean_intra_corr`
are real correlated clusters, ready for `--stage oracle --interval 1h --group_id
<id>`. A run where every group is a singleton with `mean_intra_corr IS NULL` (as
observed above) means no pairs cleared `min_overlap` — check the bug note above
before assuming this reflects real market structure.

## Verification performed (2026-08-20)

- `1d` correlation data untouched: `runs` table's `stage='correlation' AND
  interval='1d'` count was 13 before and 13 after this run.
- Command exits 0 either way (bug above does not crash the stage, it silently
  under-produces).

See also: [hourly-universe-screen.md](hourly-universe-screen.md) (prerequisite
stage) and [hourly-oracle.md](hourly-oracle.md) (next stage, once written).
