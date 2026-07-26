# Signals validation (paper-trading)

Replay ~6 months of historical `kairos_signals.py` reports day-by-day through
Phantom Ledger, a paper-trading ledger library (separate sibling package,
`phantom`), and compare the resulting realized performance against what the
signals report *claims* it should be. This is the monthly proof point that
the live advice stream is safe to commit real capital to — see
[`roadmap/phase-4-paper-trading.md`](../../roadmap/phase-4-paper-trading.md)
Task 4.1, which this playbook implements.

## Prerequisites

- GPU available: the script generates one `kairos_signals.py` report per bar
  of `--interval`, and each report is a real Kronos inference run.
- A `viability_report` run exists in `data/pipeline_results.db` for the
  chosen `--interval`, covering the full replay window (see
  [daily-signals.md](daily-signals.md)'s prerequisites — same requirement,
  just needed for every historical day in the window, not only "today").
- The `phantom` package (Phantom Ledger) is installed and importable. It
  keeps its own SQLite state, isolated from Kairos's own
  `data/pipeline_results.db` (see `--phantom-data-dir` below).

## Steps

```bash
uv run strategy/kairos_papertrade.py --months-back 6 --interval 1d --html
```

Start with a short window to sanity-check before committing to a full run
(see "Cost note" below):

```bash
uv run strategy/kairos_papertrade.py --months-back 0.5 --interval 1d
```

## When to run

Monthly. This is a validation routine, not a daily/hourly one: it exists to
confirm — on a recurring cadence — that the live signals report's expected
performance (EV, Sharpe, win rate) still holds up when replayed through a
real accounting ledger, before any real capital follows the report's advice.

## What it reads

Nothing pre-existing beyond the same `viability_report` data that
`kairos_signals.py` itself reads (see [daily-signals.md](daily-signals.md)
"What it reads"). `kairos_papertrade.py` does not read an existing signals
report — it generates its own sequence of historical reports by stepping
`kairos_signals.py` backward bar-by-bar from "now" (or `--effective_per`)
for `--months-back` months, at `--interval` granularity.

Because `kairos_signals.py`'s backward-stepping is calendar-day based, not
trading-day aware, weekend/holiday iterations resolve to the same effective
trading day as the preceding session (e.g. both a Saturday and Sunday
iteration collapse onto the prior Friday's report). `kairos_papertrade.py`
de-duplicates these before replay, so each effective trading day is played
exactly once.

## Flags

- `--db PATH` — path to `pipeline_results.db` (passthrough to
  `kairos_signals.py`; default: same default as `kairos_signals.py`).
- `--out PATH` — output dir for the final JSON/HTML reports (default:
  `results/`).
- `--months-back FLOAT` — how far back to replay, in months (default: `6`).
- `--interval STR` — bar interval for generated reports (default: `1d`).
- `--top-n INT` — max new positions opened per report day (default: `3`).
- `--capital FLOAT` — starting paper-trading account capital, EUR (default:
  `200.0`).
- `--broker STR` — Phantom Ledger broker profile to simulate (default:
  `IBKR` — supports both stock and CFD instrument types plus short
  positions, needed for futures/crypto/short signals).
- `--base-only` / `--include-finetuned` — whether to use only the base model
  (opt-in via `--base-only`) or include accepted finetuned-model overlay
  signals (default: on — finetuned overlay is used when an accepted model
  exists).
- `--min_ev_pct FLOAT` — passthrough to `kairos_signals.py`'s `run()`
  (default: `0.10`).
- `--cluster_map PATH` — optional passthrough to the allocation config's
  cluster map.
- `--phantom-data-dir PATH` — where Phantom Ledger's own SQLite DB/state
  lives (default: `data/phantom_ledger` — kept isolated from Kairos's own
  `data/pipeline_results.db`).
- `--html` — also emit an HTML report with an equity curve chart (in
  addition to the always-produced JSON report).
- `--effective_per STR` — optional override for "now" (end of the replay
  window), same format as `kairos_signals.py`'s own flag: `'YYYYMMDD
  [HHnn]'`.
- `--account-name STR` — optional override for the Phantom Ledger account
  name (default: auto-generated, e.g. `kairos_papertrade_<stamp>`).

## What it does

1. Generates one `kairos_signals.py` report per bar of `--interval`, walking
   backward from "now" (or `--effective_per`) for `--months-back` months,
   using the accepted finetuned-model overlay by default (when available),
   or using only the base model if `--base-only` is passed.
   De-duplicates reports that resolve to the same effective trading day (see
   "What it reads" above).
2. Opens one Phantom Ledger paper-trading account of type `"algorithm"`,
   tagged with an algorithm_id/version identifying this as a base-model
   `kairos_signals` validation run, funded with `--capital`.
3. Replays the de-duplicated reports oldest-first. Each report's top
   candidates are selected via the existing `strategy/allocation.py`
   allocation engine — the same engine that builds the report's own
   "Portfolio Allocation" section (see
   [`docs/rfc_allocation_sheet.md`](../rfc_allocation_sheet.md)) — configured
   with `top_k=<--top-n>` and with any ticker excluded if a paper position is
   already open for it. Allocation percentages are rescaled so the selected
   trades use up to 100% of currently-available cash (not total equity,
   since some may already be committed to open positions).
4. Selected trades are recorded as pending market orders and executed at the
   *next* report day's opening price — not the same day, since a report
   already reflects that day's closing information and executing on it would
   be lookahead bias. This matches the "executed at next-bar open" behavior
   called for in
   [`roadmap/phase-4-paper-trading.md`](../../roadmap/phase-4-paper-trading.md).
   Each position carries the stop-loss/take-profit levels from its
   originating signal; Phantom Ledger's own simulation evaluates those (and
   any other exit trigger) automatically, bar by bar, as the replay
   advances.
5. At the end of the window, any position still open is force-closed at the
   last available price, so the final trade-count/win-rate/profit-per-trade
   figures reflect only completed trades.

### Cost note

GPU inference cost scales with the number of *distinct trading days*
replayed — roughly 126 trading days for a 6-month window at `1d` interval.
Run a short window first (`--months-back 0.5`) to sanity-check the setup
before committing to a full 6-month run.

## Output

Always writes a JSON report to `--out` (default `results/`):

```json
{
  "total_profit_eur": 18.42,
  "pct_profit": 9.21,
  "pct_profit_per_trade": 0.87,
  "pct_max_drawdown": 4.6,
  "sharpe": 1.12,
  "num_trades": 21,
  "window_start": "2026-01-23T00:00",
  "window_end": "2026-07-23T00:00",
  "interval": "1d",
  "capital": 200.0,
  "broker": "IBKR",
  "base_only": true
}
```

`pct_max_drawdown` is computed over **total equity** — cash plus the value
of open positions combined — not as two separate cash/position-value
drawdown numbers.

With `--html`, also writes
`kairos_signals_papertrade_{end:%Y%m%d%H%M}_{start:%Y%m%d%H%M}_{interval}_{months}m.html`
containing:

- A title and a human-readable description of what was backtested (window,
  interval, capital, broker, base-only/finetuned mode).
- The same metrics as a table.
- An equity graph: a blue line for total equity, a green line for
  available/cash equity, and small red circles marking each position's
  open/close points on the total-equity line, connected by a gray dotted
  line, with hover tooltips showing per-trade order info.

## Empty-report troubleshooting

A day with no viable signals is expected, not an error — same framing as
[daily-signals.md](daily-signals.md)'s empty-report guidance. It simply
contributes no new trades that day; the ledger loop still advances and still
evaluates exits for any already-open positions. If *every* day in the window
comes back empty, treat it the same as an empty `kairos_signals.py` report:
check for a missing `viability_report` run covering the window, or an
`--min_ev_pct` floor that's too strict for the period (see
[daily-signals.md](daily-signals.md#empty-report-troubleshooting)).

## Automation opportunities

- A monthly cron/systemd timer, instead of running this by hand.
- Feeding the JSON report's `sharpe`/`pct_profit` straight into the Phase 4.2
  live-vs-backtest drift monitor (not yet built) instead of eyeballing it —
  see [`roadmap/phase-4-paper-trading.md`](../../roadmap/phase-4-paper-trading.md).
- Archiving successive monthly JSON reports so drift over time is visible
  without re-running old windows.

See also:
[`roadmap/phase-4-paper-trading.md`](../../roadmap/phase-4-paper-trading.md),
[`docs/rfc_allocation_sheet.md`](../rfc_allocation_sheet.md),
[daily-signals.md](daily-signals.md).
