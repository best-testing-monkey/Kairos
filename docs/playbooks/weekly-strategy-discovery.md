# Weekly strategy discovery

Find which strategies are actually effective right now, per interval, and
refresh the viability report that `kairos_signals.py` reads. Run this once a
week. Full stage-by-stage reference: [`strategy/PIPELINE.md`](../../strategy/PIPELINE.md).

## Prerequisites

- GPU available for the `base` stage (real Kronos model). The `universe`,
  `correlation`, and `oracle` stages need no GPU.
- Enough wall-clock time: the 1h/3m run is the long one (see "Runtime
  expectations" below).

## Steps

Run two `--stage auto` passes — one per interval. The first refreshes the
universe/correlation groups; the second reuses them:

```bash
# Run 1: day interval, 6-month backtest (universe + correlation refresh included)
uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1d --backtest_period 6m

# Run 2: hour interval, 3-month backtest, reusing the fresh universe/correlation
uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1h --backtest_period 3m --skip_universe
```

Each run chains universe → correlation → oracle → base per discovered group
and writes a `viability_report` row set to `data/pipeline_results.db` plus a
CSV snapshot to `results/auto_viability_report_<timestamp>.csv`. Because each
run covers one interval, you get two separate `viability_report` runs — one
for `1d`, one for `1h`. `kairos_signals.py` selects the latest run **per
interval**, so both runs coexist and neither hides the other.

## What "oracle-effective" means

The **oracle** stage runs `strategy/kairos_strategies.py` as a subprocess
with `--no-prediction`: every strategy is scored against the *actual*
next-bar OHLCV instead of a model prediction — a perfect-foresight ceiling,
no GPU needed (`strategy/kairos_pipeline.py:721-832`). The per-strategy
effectiveness metric is **shadow Sharpe** (`payload["shadow_performance"][strategy]["sharpe"]`):
every signal a strategy would have emitted, scored against the real outcome,
independent of whether it was actually ranked/executed that day. The **base**
stage repeats the same subprocess call *without* `--no-prediction`, using the
real Kronos base model — this is GPU-bound and is what confirms the oracle
ceiling holds up against actual model predictions.

## Viability criteria

A strategy is marked `viable=1` in the report only if all three hold:

- `oracle_sharpe > min_sharpe`
- `base_sharpe > min_sharpe`
- `min(oracle_signals, base_signals) >= min_signals`

Defaults: `--min_sharpe 0.0`, `--min_signals 3`. Tune with those flags if you
want a stricter or looser bar.

## Resumability

- `--force` — re-run stages even if results already exist for that
  (assets, interval, backtest_period) combination.
- `--report_only` — skip execution entirely and rebuild the viability report
  from whatever is already in the DB (useful after a partial/crashed run).
- `--skip_universe` — reuse the latest universe/correlation runs instead of
  re-screening (used above for the second, hourly pass).

## Unattended / overnight runs

Set `KAIROS_GPU_ALLOW_REBOOT=1` to permit the L4 reboot+resume step of the
GPU recovery ladder for unattended runs. A subprocess that exits `75`
(GPU healed but the current process still can't see it) is retried once
automatically by `run_backtest_subprocess`. Full ladder details (L0–L4) are
in [`CLAUDE.md`](../../CLAUDE.md) under "GPU recovery" — don't duplicate them
here.

## Runtime expectations

Base-stage backtests run at roughly 0.3–0.4 s/iteration on GPU. The 1h/3m run
covers ~2,200 bars per group and takes **at least an hour** end to end —
schedule it accordingly (start it and check back later rather than watching
it). The 1d/6m run is much shorter (~180 bars per group).

### Quick smoke test (1h / 7 days)

To verify the hourly toolchain without committing to the full run, use a
7-day window first:

```bash
uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1h --backtest_period 7d --skip_universe
```

Caveat: this writes its own `viability_report` run for `1h`, and
`kairos_signals.py` reads the **latest** run per interval — so until the real
1h/3m run completes, hourly signals would be based on the 7-day report (and
with only ~168 bars, most strategies won't reach `min_signals`). Run the
smoke test first, then the full 3m run to supersede it.

## Post-run review checklist

1. Open `results/auto_viability_report_<timestamp>.csv` (or query the
   `viability_report` table) for the run you just produced.
2. Sort by `oracle_sharpe`. Strategies with **negative oracle Sharpe** are
   candidates for the hand-curated `_DISABLED_BY_PROFILE` dict in
   `strategy/kairos_strategies.py` (~line 558) — add a
   `(interval, "SYM1,SYM2,...")` entry to silence them for that profile.
   Don't disable on a single low-`signal_count` result; corroborate with
   another window or asset set first.
3. Confirm both interval runs (1d and 1h) landed a viability row set before
   moving to the daily/hourly signals playbooks.

## Automation opportunities

- A systemd user timer (or a `/schedule` cloud routine) for the weekly pair
  of commands above — nothing is scheduled today.
- Auto-append negative-oracle strategies to `_DISABLED_BY_PROFILE` instead of
  hand-editing the dict after each review.
- Auto-prune stale `viability_report` runs / old `results/auto_viability_report_*.csv`
  snapshots once superseded.

See also: [daily-signals.md](daily-signals.md), [hourly-signals.md](hourly-signals.md).
