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
`--gsheets`/`--xlsx`/`--ods`, `--cluster_map`, `--effective_per`,
`--bars_backtest`) and output format, see
[daily-signals.md](daily-signals.md) — it's identical for `1h`.

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
  (`strategy/kairos_strategies.py:693`) resolves a different disabled set per
  `(interval, assets)` profile, so don't assume the same strategies that fire
  on daily bars will fire hourly, or vice versa.
- **This is the strongest case for automation in the whole system:** running
  this by hand every hour is impractical. If you automate one thing first,
  automate this one (see below).

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
