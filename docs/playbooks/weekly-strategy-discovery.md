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

Oracle always evaluates the **full** strategy suite, including strategies
currently disabled for this profile (it passes `--no_disabled_filter`) —
that's what lets a strategy earn its way back in automatically once its
performance turns positive (see "Post-run review checklist" below). `base`
still skips disabled strategies, since there's no point spending GPU time
confirming a strategy already known non-viable.

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

> **Known issue (2026-07-18):** unattended recovery is currently unreliable —
> the L3 step fails when restarting the window manager. Until that's fixed,
> treat long GPU runs as needing a healthy GPU throughout; the ladder is not
> a dependable safety net.

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
2. Review the `[disabled]` diff each oracle run prints to stdout (newly
   disabled / re-enabled strategy names), or query the `disabled_strategies`
   table directly, or open the CSV mirror written to
   `results/oracle_disabled_strategies_<timestamp>.csv`. Disabling is fully
   automatic per `(interval, assets)` profile: a strategy is disabled when
   its oracle `avg_pnl_per_trade < 0` and `signal_count >= 5` (tune the
   threshold with `--disable_min_signals`), and re-enabled the moment it no
   longer meets that bar — there's nothing to hand-edit. If you change
   `--disable_min_signals` after the fact, run `--stage rebuild_disabled` to
   recompute the whole table from existing oracle results instead of
   waiting for the next weekly oracle run.
3. Confirm both interval runs (1d and 1h) landed a viability row set before
   moving to the daily/hourly signals playbooks.

## Automation opportunities

- A systemd user timer (or a `/schedule` cloud routine) for the weekly pair
  of commands above — nothing is scheduled today.
- The per-profile `disabled_strategies` table is now auto-maintained (see
  "Post-run review checklist" above); the remaining hand-curated artifact is
  the coarser `_DISABLED_BY_CLASS` per-`(interval, asset_class)` fallback
  table in `strategy/kairos_strategies.py`, used only for profiles that have
  never been oracle-tested. Automating that is still future work.
- Auto-prune stale `viability_report` runs / old `results/auto_viability_report_*.csv`
  snapshots once superseded.

See also: [daily-signals.md](daily-signals.md), [hourly-signals.md](hourly-signals.md).
