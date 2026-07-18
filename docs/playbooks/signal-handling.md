# Signal handling

What to do once a `kairos_signals.py` report exists. Everything downstream of
the report is manual today — there is no broker integration. Full sizing
reference: [`docs/rfc_allocation_sheet.md`](../rfc_allocation_sheet.md);
implementation: `strategy/allocation.py`.

## Prerequisites

- A signals report (`results/kairos_signals_<stamp>.md`) and, if you ran with
  `--xlsx`/`--ods`/`--gsheets`, the companion spreadsheet with `strategies`,
  `signals`, and `Allocation` tabs.

## Steps

1. **Open the report and sanity-check signals.** Direction conflicts and
   duplicate-asset rows are already collapsed before you see them. Each
   candidate passed through gates in this order (`strategy/allocation.py:301`):
   `SCHEMA_ERROR` → `DISABLED` → `LOW_N` (fewer than `min_n=50` backtest
   trades) → `NEG_EV_NET` (negative expected value net of costs). Rows that
   failed a gate don't appear as tradeable signals — check the spreadsheet's
   non-selected rows if you want to see why something was excluded.

2. **Understand how sizing was computed.** Survivors are ranked by score and
   the top 12 are kept. Sizing: shrink EV net of round-trip cost, apply
   fractional Kelly (multiplier `0.35`), then cap per position at 15%
   (`max_pos_pct`), cap per correlation cluster at 25% (`max_cluster_pct` —
   needs a `--cluster_map` CSV passed to `kairos_signals.py`), cap total gross
   exposure at 100% (`gross_cap_pct`), and zero out anything left under a 1%
   dust filter (`dust_min_pct`) — dust is a single-pass zero, not
   redistributed (`strategy/allocation.py:52-58,462-560`).

3. **Use the spreadsheet's Enabled column as your veto.** The Allocation tab
   has a user-editable **Enabled** column (column A). Unchecking a row zeroes
   that position and live-renormalizes the remaining enabled rows back to
   100% inside the spreadsheet — this is the intended human veto step
   (formulas added in commit `4ecac82`). Use it to drop a signal you disagree
   with without re-running the pipeline.

4. **Place orders manually.** Use the entry/stop/target from each signal row.
   The plain-English advice sentence in the `## Signals` section already
   states SL/TP in both price and percent terms. Treat FLAT or zero-size rows
   as "no action" — there's nothing to place.

5. **Keep a record.** The timestamped report file is your audit trail — keep
   it, along with the spreadsheet showing which rows you enabled/disabled.
   Note any deviation from what the report recommended (different size,
   skipped a signal, etc.) so you can review it later against outcomes.

6. **When to distrust a report.** Treat it with more caution (or skip it)
   when: a strategy is flagged by a regime/decay filter (e.g. the
   `EconCalendarGuardStrategy` wrapper vetoes new entries ahead of a
   high-impact economic event); the weekly discovery run behind it is stale
   (viability run older than roughly two weeks — rerun
   [weekly-strategy-discovery.md](weekly-strategy-discovery.md) first); or the
   report's `## Failures` footer shows fetch/prediction errors for some of
   the assets you care about.

## Automation opportunities

- A paper-trading ledger (roadmap Phase 4, unbuilt) to track hypothetical
  fills against real outcomes before committing real capital.
- Order-ticket generation directly from the Allocation sheet (entry/stop/
  target/size formatted for manual entry into a broker UI).
- Position/PnL reconciliation against the recorded reports.
- Eventual exchange execution via ccxt (roadmap Phase 5, unbuilt) — see
  [`ROADMAP.md`](../../ROADMAP.md).

See also: [daily-signals.md](daily-signals.md), [hourly-signals.md](hourly-signals.md).
