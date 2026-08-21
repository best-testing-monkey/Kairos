# 1h discovery sweep: finding profitable signal/instrument combinations

**This is not a playbook for something already done — it's directions for a
piece of work that hasn't been run yet.** Everything else in
`docs/playbooks/hourly-*.md` documents individual pipeline stages that were
each *verified* against `1h` this session (universe, correlation, oracle,
base, finetuning, signals, papertrade). None of those verification runs were
a strategy-discovery pass — each deliberately used a tiny window (a few days,
one asset group) to keep the check fast. This document is about the actual
job: finding which `(strategy, asset)` combinations have a real, persistent
edge at `1h`, the way the `1d` side of this codebase already has (128
finetuned `1d` checkpoints, months of `oracle`/`base` sweeps across the full
candidate universe — see `strategy/PIPELINE.md`'s "Storage" section for the
scale). `1h` currently has 5 finetuned checkpoints and has only ever been
pointed at one group (`ZW=F`) end-to-end.

## Why this is a separate piece of work, not a byproduct of E10–E17

The multi-interval rollout (`docs/tickets/DESIGN_DOC_multi_interval_1h.md`,
epics E10–E17) built and verified the **mechanism** — universe screening,
correlation grouping, the oracle/base/finetuned funnel, signal generation,
papertrade execution, all working correctly at `1h`. It answers "does the
plumbing work at this interval." It does not answer "which `1h` strategies
are actually profitable," because none of E10–E17's live-verification runs
swept more than one asset group. A 3-4 day, 5-asset, 100%-stop-loss
papertrade run (E16-S02) is proof the MTM/financing guard fires correctly —
it is not, and was never meant to be, a performance verdict.

## Prerequisite: check BUG-04's residual finding first

`docs/todo.md`'s BUG-04 entry notes that after the upstream price_cache DST
fix landed, most crypto symbols still report `$vol=0.0` in `1h` universe
screening despite having real bar/ATR data — flagged as "not yet
investigated." A full sweep run against the current universe would still be
crypto-starved if this is actually suppressing real survivors. Before
committing real GPU time to a sweep:

1. Run `uv run ./strategy/kairos_pipeline.py --stage universe --interval 1h`
   and check how many symbols pass, and whether crypto specifically is still
   near-zero.
2. If it's still thin, spend an hour tracing `$vol=0.0`'s root cause
   (`compute_universe_stats`'s `dollar_volume` computation, or possibly
   another price_cache-side gap in the same family as BUG-04) before
   sinking a multi-hour sweep into a universe that's mostly excluded by a
   fixable bug.
3. If crypto now passes at healthy numbers, proceed — the sweep is worth
   running as-is.

## What "the sweep" actually is

`--stage auto` is the existing tool built for exactly this — it already
chains universe → correlation → oracle → base per interval, across however
many asset groups correlation discovers, and builds a consolidated
**viability report**. Full mechanical reference: `strategy/PIPELINE.md`'s
"Stage auto: Unified discovery pipeline" section (flags, resumability,
per-run prediction caching, the `disabled_strategies` auto-maintenance, the
viability report schema) — this document doesn't repeat that, it adds the
`1h`-specific framing PIPELINE.md doesn't cover.

```bash
uv run ./strategy/kairos_pipeline.py --stage auto \
    --intervals 1h \
    --backtest_period 3m \
    --asset_class crypto
```

Start scoped (one `--asset_class` at a time, a shorter `--backtest_period`
like `3m` before committing to `6m`), not the whole candidate universe in one
run — see "Scope and cost" below for why.

## Interpreting results toward "profitable and persistent"

A row in the `viability_report` table (or its CSV dump) being `viable=True`
means `oracle_sharpe` and `base_sharpe` both cleared `--min_sharpe` with
enough signals — that's evidence of edge in *this one backtest window*, not
evidence of *persistent* edge. Three things worth doing before trusting a
result:

- **Respect the small-`signal_count` caveat** (`PIPELINE.md` §5): a strategy
  with `n < 3` signals has a statistically meaningless Sharpe. Don't act on
  a single thin-sample result — extend `--backtest_period` or corroborate
  against a second asset set/window first.
- **Check persistence across time, not just one window.** A strategy viable
  on one 3-month `1h` window could be a regime-specific fluke. The cheapest
  way to check whether an edge holds up over a longer span without
  re-running GPU inference repeatedly is `strategy/kairos_signal_replay.py`
  (`--precompute` once, then `--replay` with different windows/selection
  rules in seconds — see its own module docstring and
  `docs/tickets/DESIGN_DOC_offline_signal_replay.md` for how it works).
  It's unleveraged-only and its cost model is simplified relative to live
  papertrade, so treat it as a fast filter for "worth a real papertrade
  check," not a final answer.
- **Confirm with a live papertrade run once a shortlist exists.** Once
  `--stage auto` + replay narrows things down to a specific set of
  `(strategy, asset)` combinations worth trusting, `kairos_papertrade.py
  --interval 1h` over a longer real window (not the 3-4 day smoke-test
  window E16-S02 used) is the final, most realistic check — it's the only
  one that exercises the actual MTM/margin/financing machinery this
  session's E16 epic verified.

## Scope and cost — don't try to do this in one shot

The `1d` side of this pipeline reached its current state (128 finetuned
checkpoints, a mature `disabled_strategies` table, months of accumulated
`oracle_results`/`model_results`) over many separate sessions, not one run.
`1h` has 24x the bars-per-day of `1d`, so oracle/base backtests over
comparable calendar windows take proportionally longer per group. Sensible
incremental approach:

1. One `--asset_class` at a time (`crypto` first is a reasonable choice —
   it's the class BUG-04 most affected, so also doubles as confirmation the
   prerequisite fix actually worked at scale).
2. A shorter `--backtest_period` (`1m`–`3m`) for the first pass; extend once
   you see which groups are worth a longer look.
3. `--stage auto` is resumable (`PIPELINE.md`'s "Resumability and
   `--force`" section) — a crashed or interrupted run can just be re-invoked
   with the same command and it'll skip whatever already completed. No need
   to babysit a single unbroken multi-hour run.
4. `finetune_next` (already fixed for cross-interval candidacy this session,
   E14-S02) is the natural follow-up once `--stage auto` has real `1h`
   `oracle`/`base` viability data to pick candidates from — it selects and
   trains automatically, one candidate per invocation.

## Where this fits

This is deliberately **not** added as a new epic in `docs/todo.md`'s
multi-interval rollout section — E10–E17 was about the mechanism being
correct, and it's done. This document is the on-ramp to the next, larger,
open-ended piece of work: actually using that mechanism at scale to find
real `1h` edge. Track it separately, sized in whatever increments make sense
once the prerequisite check above is done.
