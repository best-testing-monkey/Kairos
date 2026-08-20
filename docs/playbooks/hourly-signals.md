# Hourly signals

Generate an actionable signals report for hourly (`1h`) bars. Same mechanics
as [daily-signals.md](daily-signals.md); this playbook covers only what
differs for the hourly cadence. Requires
[weekly-strategy-discovery.md](weekly-strategy-discovery.md) to have
completed at least once for the `1h` interval.

## Prerequisites

- GPU available (same batched-prediction requirement as daily).
- A `viability_report` run exists for `1h` in `data/pipeline_results.db`.

## Steps

```bash
uv run ./strategy/kairos_signals.py --intervals 1h --xlsx
```

Run this a few minutes past the top of the hour — `fetch_data_raw` rounds
down to the last closed bar, so running too early just repeats the previous
hour's bar.

For the full flag reference (`--min_ev_pct`, `--pred_samples`, `--all`,
`--gsheets`/`--xlsx`/`--ods`, `--cluster_map`, `--base_only`,
`--effective_per`, `--bars_backtest`) and output format — including the
`model` column and the automatic accepted-finetuned-model overlay/
`base_shadow` comparison tab — see [daily-signals.md](daily-signals.md), the
"Finetuned models" section — it's identical for `1h`.

## Hourly-specific caveats

- **Data quality:** yfinance caps 1h history at 729 days and is "marginal
  for hourly" per [`ROADMAP.md`](../../ROADMAP.md) — expect some delay/gaps
  relative to a real-time feed. A ccxt-based migration is roadmap Phase 5,
  not yet built.
- **EV floor matters more:** hourly signal expected values are smaller in
  absolute terms than daily ones, so `--min_ev_pct` (default `0.10`) is more
  likely to bind relative to round-trip trading costs. Check the `## Skipped`
  footer if the report looks thin.
- **Disabled-strategy set differs from `1d`:** `resolve_disabled_strategies`
  resolves a different disabled set per `(interval, assets)` profile —
  DB-backed via the auto-maintained `disabled_strategies` SQLite table for
  profiles that have been oracle-tested, falling back to the hand-curated
  `_DISABLED_BY_CLASS` table only for profiles that haven't — so don't
  assume the same strategies that fire on daily bars will fire hourly, or
  vice versa.
- **This is the strongest case for automation in the whole system:** running
  this by hand every hour is impractical. If you automate one thing first,
  automate this one (see below).
- **Live-verified cache behavior (E15-S01, 2026-08-20):** ran three times
  across a real hour boundary against a populated `1h` pipeline (`ZW=F`/
  `ZEC-USD` finetuned, several base-model groups). Run 1 (fresh):
  `signals_cache` had 44 `interval='1h'` rows after. Run 2 (same clock hour,
  ~2 min later): report was byte-identical to run 1 and `signals_cache` row
  count stayed at 44 — genuine cache hit (`user` CPU time dropped from
  ~10.5s to ~5.5s; wall-clock `real` time is noisy run-to-run because model
  weights reload from disk every subprocess invocation regardless of
  `signals_cache` state, so don't use wall-clock alone to judge a hit).
  Run 3 (after the real hour boundary): row count grew to 88 and the report
  content genuinely changed (new EV%/prices per signal, one strategy —
  `fade_extreme/BCH-USD` — that hadn't cleared the EV floor in runs 1-2
  showed up fresh). This is the live-conditions confirmation of the E0
  `_cache_as_of_value` fix (bar-floored cache key for intraday intervals):
  it correctly reuses within an hour and correctly busts across one.
- **`## Skipped` and `## Failures` footers read as expected, not broken:**
  on this same run, `Skipped` entries were legitimate (`zero-size signal
  dropped (no Kelly edge)`, `ev_pct below threshold`) — no "unknown
  strategy" mass-skip, so `resolve_disabled_strategies`/strategy registry
  resolution works correctly for `1h`. `Failures` showed several equities
  (`CB`, `CRM`, `DE`, `ABT`, `ABBV`, `AAPL`, `BA`, `COST`) failing with
  `Not enough data ... need 300 bars, got 259` — a real, current data-
  availability limit for those symbols at `1h` (not a bug in this pipeline),
  worth knowing before assuming every asset class is equally ready.

## Automation opportunities

- Highest-value automation candidate here: a loop/cron job that runs this
  every hour with notification-on-signal-only (don't page for an empty or
  unchanged report).
- Same Telegram/scheduling gap as daily — roadmap Phase 2 (`kairos_live.py`)
  and Phase 3 (scheduling & delivery) are both unbuilt; see
  [`ROADMAP.md`](../../ROADMAP.md).
- A ccxt-backed data source (roadmap Phase 5) would remove the yfinance
  729-day/marginal-quality caveat above.

See also: [signal-handling.md](signal-handling.md).
