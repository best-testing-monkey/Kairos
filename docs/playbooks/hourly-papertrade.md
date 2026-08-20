# Hourly papertrade

Run the full `kairos_papertrade.py` loop (signal generation → allocation →
Phantom Ledger fills → MTM/financing accrual) against `1h` price data — the
end-to-end simulation that exercises every earlier `--interval 1h` stage
together. Unlike `1d` papertrade, an `1h` window has **24x more loop
iterations per calendar day** for the same wall-clock span, so the day-level
MTM/financing accrual is guarded (E16-S01, commit `65de700`) to fire at most
once per calendar date regardless of how fine the loop step is — this
playbook documents the live-conditions proof that guard actually works.

## Prerequisites

- Universe → correlation → oracle → base → (optional) finetuned → signals
  must all have run for `1h` first (see the other `hourly-*.md` playbooks).
  This run used the existing `1h` pipeline data as-is (`ZW=F`/`ZEC-USD`
  finetuned models, `signals_cache` populated from E15-S01).
- **GPU required** — real Kronos model inference, same as base/finetuned.
- **Start with a SHORT `--months-back`** for a first `1h` run. `0.1` months
  (~3 calendar days ≈ 70+ hourly bars) was enough to prove the guard across
  multiple day boundaries and still completed in a few minutes once the
  prediction cache was warm from earlier runs this session; a naive `6`
  (the `1d` default) would mean ~24x the iterations of an equivalent `1d`
  run for the same window.

## Steps

```bash
uv run ./strategy/kairos_papertrade.py --interval 1h --months-back 0.1 \
  --capital 1000 --no-telegram --db data/pipeline_results.db
```

Add `--max-leverage <N> --margin-utilization <frac>` (N > 1.0) to exercise
margin/financing math — the default `--max-leverage 1.0` is cash-only, and
`daily_financing()` is a no-op with no borrowed capital, so a cash-only run
alone doesn't exercise the accrual path this story cares about.

## What was observed (2026-08-20/21, two runs)

**Run 1 — cash-only (`--max-leverage 1.0`, the default), window
2026-08-17→2026-08-20 (~70 hourly iterations across 3 calendar dates):**
exit 0, `total_profit_eur=-52.06`, `num_trades=46`.

**Run 2 — leveraged (`--max-leverage 3.0 --margin-utilization 0.8`), window
2026-08-18→2026-08-21 (4 calendar dates):** exit 0, `total_profit_eur=-82.04`,
`num_trades=46`.

### The core check: one `kairos_mtm_daily` row per calendar date, not per iteration

`kairos_mtm_daily` lives in Phantom's own DB
(`data/phantom_ledger/phantom.db`, **not** `pipeline_results.db` —
`_ensure_mtm_daily_table`/`_insert_mtm_daily_row` both use `client._conn`),
keyed by `account_name` (one fresh timestamped account per papertrade
invocation). Filtered to each run's own `account_name`:

```
-- Run 1 (kairos_papertrade_202608202349), ~70 hourly iterations, 3 dates:
2026-08-18|1
2026-08-19|1
2026-08-20|1

-- Run 2 (kairos_papertrade_202608210001), ~90 hourly iterations, 4 dates:
2026-08-18|1
2026-08-19|1
2026-08-20|1
2026-08-21|1
```

Exactly one row per distinct calendar date in both runs — not one row per
hourly iteration (which would have been ~24-70+ rows for these windows).
This is the direct live-conditions proof of E16-S01's guard.

### Financing sanity check — a real limitation of this test window

Both runs' `kairos_mtm_daily` rows show `financing_accrued_day=0.0` and
`gross_notional=0.0` on every date, even the leveraged run. This is
**expected, not a bug**: `daily_financing()` only charges positions that are
still **open at day-close** (see `kairos_mtm.py`'s own docstring: "the entry
day counts, the exit day does not"). With `top_n=3` and a thin `1h` signal
set over a short window, this run's 46 trades apparently all opened and
closed intraday (same-day round trips) rather than surviving to a daily
close snapshot — so there was never a nonzero financing charge to sanity-check
against a hand-calculated `/360` expectation in either run.

This does NOT weaken the guard verification above (the row-count proof holds
regardless of dollar amounts), but it means this specific pair of runs
couldn't positively demonstrate "not 24x over-accrued" with real nonzero
numbers. What DOES cover that: `financing_day`'s computation and
`corrected_cash -= financing_day` both live inside the exact same
`if last_financing_date != day_start.date():` guard block as the
`kairos_mtm_daily` write (confirmed by reading the code, `kairos_papertrade.py`
~line 2308-2432) — since the write is proven to fire once per date, financing
computation necessarily does too, in this same code path. E16-S01's own unit
tests (`tests/unit/test_kairos_papertrade_financing_guard.py`) separately
prove the call-count behavior with mocked, nonzero-returning
`compute_daily_financing_total()`. A future run with positions that
genuinely survive a day-close (larger `--capital`/`--top-n`, longer window,
or a deliberately long-held position) would be a good target for a live
nonzero spot-check, but wasn't produced by this verification pass.

### Watchdog log

`data/papertrade_watchdog.log` — no `slow`/`crash`/`error`/`traceback`
entries for either run's PID (checked by grepping the specific PIDs each run
printed, `3012522` and `3020400`). Per-group timings were all well under
`_SLOW_ITERATION_THRESHOLD_SECONDS` (0.2-4.2s per group, threshold is 60s).

### A pre-existing, documented warning (not caused by this story)

Run 2 (leveraged) printed:
```
WARNING: cash reconciliation gap of 1035.51 EUR between phantom's raw
account.cash (917.96) and Kairos's day-loop corrected_cash + open
unrealized P&L (-117.55). See docs/tickets/DESIGN_DOC_mtm_margin_leverage.md
Section 4.2.
```
This is an existing, documented diagnostic in the codebase (references its
own design-doc section), not something E16-S01 introduced — flagging it here
because it fired on this short/thin `1h` window and is worth knowing about
if you see it on a future run, but investigating it further is out of this
story's scope.

## Caveats

- `1h` universe screening still only has a handful of real survivors (see
  BUG-04's residual `$vol=0.0` finding) — this run's asset universe
  (`ZEC-USD`, `HBAR-USD`, `DASH-USD`, `JTO-USD`, `UNI7083-USD`, `BCH-USD`,
  `SOL-USD`, `XLM-USD`) is thin and crypto-only, not representative of a
  fully-populated pipeline.
- Prediction-cache prewarm's first model load/switch is the dominant cost on
  a cold cache (~50min ETA shown in `tqdm`'s initial estimate before the
  cache warmed up) — budget for that on a genuinely first `1h` papertrade
  run; subsequent runs against overlapping windows reuse the shared
  `kairos_predcache` and are much faster (run 2 finished in under 10 minutes
  against a warm cache from run 1).

See also: [hourly-signals.md](hourly-signals.md) (the signal-generation
stage this consumes) and `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md`
(the MTM/margin design this stage implements).
