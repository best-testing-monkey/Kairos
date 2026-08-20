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

## ⚠️ Known bug: `min_overlap` scaling produces zero pairs at 1h (found 2026-08-20, unfixed)

**A live run of this stage for `1h` currently produces ZERO correlated pairs — every
survivor ends up as an isolated singleton group, even when real correlation almost
certainly exists (e.g. between crude oil and gas futures).** This is not a "thin
data" edge case; it happens on every run. Root cause, confirmed by reading the code
and reproducing live:

- `run_stage_correlation`'s own price-series fetch window is `bars_needed = 400`
  (`strategy/kairos_pipeline.py`, ~line 781), **not scaled by interval** — the fetch
  window scales in *calendar days* via `calendar_days_for_bars(400, bars_per_day,
  ...)`, which for `1h` (`bars_per_day=24`) works out to only `400/24 ≈ 17` calendar
  days, i.e. roughly 400 1h bars fetched per symbol — the exact same **bar count**
  fetched for `1d` (400 daily bars), just compressed into far fewer calendar days.
- E11-S01 (commit `b082e53`) scaled `compute_pair_correlation`'s `min_overlap`
  threshold by the SAME 24x factor (`150 × BARS_PER_DAY["1h"] = 3600`), intending to
  preserve "150 calendar days of required overlap" — but never scaled the fetch
  window (`bars_needed`) to match. The result: at `1h`, the stage requires 3600
  overlapping bars but only ever fetches ~400. `overlap_bars < min_overlap` is true
  for every pair, unconditionally, so `compute_pair_correlation` returns
  `(None, None, overlap_bars)` for everything and no pair is ever scored.
- Confirmed live: a fresh `1h` universe run (see below) produced 4 survivors
  (`CL=F`, `NG=F`, `SI=F`, `ZW=F` — all `fx_commodity` futures); the correlation
  run against them produced **0 pairs, 4 singleton groups** (`run_id=729`).

**This needs a follow-up fix** (not made by this playbook/story — E11-S02 is
verification-only, no code changes) — likely either: scale `bars_needed`/the fetch
window by the same interval factor as `min_overlap`/`roll_window` (so `1h` fetches
~9600 bars, i.e. the same ~400 calendar days of history as `1d`, not just ~17 days),
or reconsider whether `min_overlap`/`roll_window` should be scaled by calendar-day
equivalence at all versus a fixed bar count tuned directly for `1h`. Until fixed,
**`--stage correlation --interval 1h` is non-functional** — it runs without
erroring, but produces no real groupings.

## Separate finding: crypto universe screening broke for 1h on this same run (2026-08-20)

Not a correlation-stage bug, but it's the reason the correlation run above only had
4 non-crypto survivors to work with. A fresh `--stage universe --interval 1h` run
(`run_id=728`) failed **every single crypto symbol except one** (`GALA-USD`, which
failed on `low_dollar_volume` instead) with:

```
fetch_error: Cannot infer dst time from 2025-11-02 01:00:00 as there are no repeated times
```

This is a pandas/timezone DST-transition error, not an E10/E11 logic bug — it
happens because the universe stage's 729-day lookback window (yfinance's 1h history
cap) now reaches back across the 2025-11-02 US DST fall-back date, and something in
the fetch/localization path chokes on the ambiguous repeated hour. An older `1h`
universe run from 2026-07-05 (`run_id=44`, 124 survivors including plenty of crypto)
predates this — its 729-day window didn't yet reach back across that DST boundary.
This will keep recurring and get worse as "today" moves further past the DST date
(the window will cross it for longer). Flagging for a separate follow-up story; not
fixed here.

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
