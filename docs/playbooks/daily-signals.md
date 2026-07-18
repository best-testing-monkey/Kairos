# Daily signals

Generate an actionable signals report for daily (`1d`) bars from the latest
weekly viability run. Requires [weekly-strategy-discovery.md](weekly-strategy-discovery.md)
to have completed at least once for the `1d` interval.

## Prerequisites

- GPU available: `kairos_signals.py` runs one batched Kronos prediction per
  `(assets, interval)` group.
- A `viability_report` run exists for `1d` in `data/pipeline_results.db`.

## Steps

```bash
uv run ./strategy/kairos_signals.py --intervals 1d --xlsx
```

## When to run

`fetch_data_raw` rounds down to the last **closed** bar, so run this any time
after the daily bar closes: after 00:00 UTC for `-USD` crypto symbols (which
trade 24/7), or after exchange close for equities/FX. Running earlier just
repeats yesterday's bar — harmless, but redundant.

## What it reads

The latest per-interval `viability_report` run for `1d` — i.e. the most
recent weekly run that covered daily bars, independent of whether an hourly
run finished more recently. Only rows with `viable=1` are used unless you
pass `--all`.

## Flags

- `--min_ev_pct` (default `0.10`) — minimum expected value as a percent of
  entry price; non-FLAT signals below this go to the `## Skipped` footer.
  Set to `0` to disable.
- `--pred_samples` (default `100`) — prediction sample count. This is a hard
  floor per `CLAUDE.md`; don't reduce it as a shortcut.
- `--all` — include non-viable rows too (default is viable-only).
- `--gsheets` / `--xlsx` / `--ods` — also write the Stats/Signals tables to a
  Google Sheet, local `.xlsx`, or local `.ods` file respectively.
- `--cluster_map` — CSV mapping ticker → cluster name, used by the Allocation
  sheet's cluster caps.
- `--effective_per "YYYYMMDD [HHnn]"` — simulate "now" as a fixed timestamp,
  for backtesting/QA the report instead of using the real current time.
- `--bars_backtest N` — generate N reports stepping backward bar-by-bar from
  `--effective_per` (or now), for QA over a historical window.

## Output

`results/kairos_signals_<YYYYMMDDHHMM>.md`, plus (with `--xlsx`/`--ods`/`--gsheets`)
a spreadsheet with `strategies`, `signals`, and `Allocation` tabs. The
markdown report has a `## Stats` table (per strategy: direction, size, entry,
stop, target, expected value, oracle/base viability stats) and a `## Signals`
section of plain-English advice sentences, e.g. *"Strategy dfa_persistence
advised **Long** position on BTC-USD for 12% liquidity with SL at 58,900.00
(-3.1%) and TP at 63,400.00 (+4.2%). Exit by TP/SL."*

Signal fields include: strategy, symbol, interval, direction, size, entry,
stop, target, expected_value, ev_pct, oracle_sharpe, base_sharpe, win rates,
and signals_per_week.

## Empty-report troubleshooting

- **No viability run for `1d`** — the weekly discovery playbook hasn't been
  run yet for that interval. Run it first.
- **No rows with `viable=1`** — every strategy failed the viability bar for
  this asset/interval combination; rerun with `--all` to see what exists and
  why it didn't qualify.
- **All signals below the EV floor** — every candidate signal's `ev_pct` is
  under `--min_ev_pct`; check the `## Skipped` footer, and consider whether
  `0.10` is too strict for the current regime.

## Automation opportunities

- A cron/systemd timer at bar close plus a small delay, instead of running
  this by hand every day.
- Delivery via Telegram or another notification channel once
  `kairos_live.py` (roadmap Phase 2, unbuilt) and the Phase 3 scheduling/
  delivery layer land — see [`ROADMAP.md`](../../ROADMAP.md).
- Auto-upload to a *fixed* Google Sheet (update in place) instead of creating
  a new sheet every run with `--gsheets`.

See also: [hourly-signals.md](hourly-signals.md), [signal-handling.md](signal-handling.md).
