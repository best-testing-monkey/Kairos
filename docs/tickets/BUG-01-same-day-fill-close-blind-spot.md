# BUG-01 — Same-day fill/close blind spot in corrected_cash / MTM day-loop bookkeeping

**Severity:** High. Found via live end-to-end `/verify` run of the MTM margin/leverage epic
(docs/tickets/DESIGN_DOC_mtm_margin_leverage.md), not caught by any of the 157 unit/regression
tests added across E4-S09..E5-S15.

## Description of bug

`strategy/kairos_papertrade.py`'s `main()` day loop maintains a Kairos-side `corrected_cash`
running ledger and drives the daily `kairos_mtm_daily` snapshot by diffing currently-open
positions against a `known_open_ids` set, **once per day, after `client.runner.backtest()`
returns** (`strategy/kairos_papertrade.py` ~lines 2082-2098):

```python
current_open = client.positions.list(account_name=account_name, status="open")
current_open_ids = {p.id for p in current_open}
for pos in current_open:
    if pos.id not in known_open_ids:
        corrected_cash += _fill_cash_delta(pos, include_notional=...)
for closed_id in known_open_ids - current_open_ids:
    closed_pos = client.positions.get(closed_id)
    corrected_cash += _close_cash_delta(closed_pos, include_notional=...)
known_open_ids = current_open_ids
```

A position that both **fills and closes within the same `runner.backtest()` call** (i.e. entry
and exit happen on the same bar/day — e.g. a stop-loss or take-profit hit on the same day as
entry, which is routine for tight stops on volatile daily-bar assets) never appears in
`current_open` at any point: by the time `current_open` is queried, the position is already
`status='closed'`. It is therefore invisible to **both** loops above — it's never in
`known_open_ids` to be detected as newly-closed, and it's never seen as a new fill either.
Neither `_fill_cash_delta` nor `_close_cash_delta` is ever applied for it, and it's excluded
from `mtm_positions` (hence every `kairos_mtm_daily` row) for the entire day it existed.

### Confirmed live reproduction

A real end-to-end run of `kairos_papertrade.py` (real Kronos base-model inference, real price
data via price_cache/yfinance fallback, `--max-leverage 2.0 --margin-utilization 0.6
--months-back 0.2 --capital 5000`) produced 3 real closed trades (WLD-USD), each with
`entry_datetime == exit_datetime` to the second (confirmed via direct SQL query against the
run's phantom DB `positions` table), for real profit:

```
total_profit_eur = 96.56043592000697   pct_profit = 1.9312087184001347   num_trades = 3
```

But every one of the 5 `kairos_mtm_daily` rows for that run showed:

```
cash=5000.0  equity=5000.0  gross_notional=0.0  initial_margin_used=0.0
```

unchanged for the entire window, and the JSON report's MTM block was all zeros:

```
mtm_total_return_pct=0.0  mtm_max_drawdown_pct=0.0  mtm_margin_utilization_peak=0.0
```

The run's own `_reconcile_cash_and_log` second check (added E4-S09) correctly flagged the
resulting divergence — `WARNING: cash reconciliation gap of 96.5604 EUR between phantom's raw
account.cash (5096.5604) and Kairos's day-loop corrected_cash + open unrealized P&L
(5000.0000)` — but nothing in the current code corrects it; the warning is log-only.

None of the existing tests exercise this path: `tests/unit/test_kairos_papertrade_leverage_regression.py`'s
synthetic fixtures deliberately construct fills on one day with SL/TP triggering on a *later*
day specifically to avoid same-day resolution (see that file's own candidate-bar comments,
e.g. `"fills at Open=100, no SL/TP touch"` on the fill day). `test_kairos_papertrade_mtm_repro.py`
replays a frozen fixture but its replay construction also processes fill and close on
whatever distinct dates the real historical positions actually used, and this specific
fixture happened not to contain a same-day round trip.

## Definition of correct functionality

Every position that is filled and/or closed during a single day-loop iteration — **including
one that both fills and closes within that same iteration** — must:

1. Have exactly one `_fill_cash_delta` and exactly one `_close_cash_delta` applied to
   `corrected_cash` (matching the existing multi-day invariant: `_fill_cash_delta(pos) +
   _close_cash_delta(pos) == compute_corrected_realized_pnl(pos)` for a spot/full-notional
   trade, or the margin-locked equivalent — see `_close_cash_delta`'s docstring).
2. Be reflected in that day's `kairos_mtm_daily` snapshot in some economically sound way (at
   minimum: the day's `corrected_cash`/`equity` must include its P&L; whether it also briefly
   contributes non-zero `gross_notional`/`initial_margin_used` for that one day is an
   implementation choice, but it must not be silently invisible).
3. After a run containing only same-day round-trip trades, `kairos_mtm_daily`'s final equity
   must still reconcile with `build_closed_trade_equity_curve`'s final equity — the same
   convergence invariant already proven for multi-day trades in
   `tests/unit/test_kairos_papertrade_mtm_repro.py::test_final_mtm_equity_equals_final_closed_trade_equity`.
   A same-day trade must not be excluded from that reconciliation.

## Reproduction instructions

**Live repro** (what was actually run to find this): requires GPU/model/network, not suitable
for a unit test. See the exact CLI invocation and confirmed root cause above.

**Suggested deterministic repro for the fix's regression test** — build on the existing mocked
`main()` harness in `tests/unit/test_kairos_papertrade_leverage_regression.py`
(`_FakeBarsProvider`, `_run_main`, `_dated_rows`/`_candidate` helpers). The key change needed
versus that file's existing scenarios: construct a bar for the FILL day whose `High`/`Low`
already crosses the candidate's `stop`/`target` on that same day, e.g.:

```python
day1 = datetime(2024, 1, 2)
bars_by_ticker = {
    "TICKA": {
        day1.date(): (100.0, 145.0, 95.0, 130.0, 1000.0),  # Open=100 (fills entry),
                                                             # High=145 >= target(140) -> same-day TP
    },
}
```

with a single candidate `_candidate("TICKA", entry=100.0, stop=90.0, target=140.0)` offered on
the day *before* `day1` (so it fills via `runner.backtest()` on `day1`). Run `main()` through
the existing harness, then assert (before the fix, this should FAIL, proving the repro is
real):

```python
row = client._conn.execute(
    "SELECT cash, equity, gross_notional FROM kairos_mtm_daily WHERE account_name=? ORDER BY date",
    (account_name,),
).fetchall()
closed = client.positions.list(account_name=account_name, status="closed")
assert closed[0].entry_datetime.date() == closed[0].exit_datetime.date()  # confirms same-day
# corrected_cash / kairos_mtm_daily must reflect the trade's P&L, not stay flat at capital
assert row[-1][1] != pytest.approx(CAPITAL)  # currently fails: stays exactly at capital
```

## Context for whoever fixes this

- `strategy/kairos_papertrade.py`: the day-loop fill/close diffing block (~lines 2075-2110),
  `_fill_cash_delta`/`_close_cash_delta`/`_use_full_notional` (~lines 864-1010),
  `_fetch_day_close_bars`/`compute_daily_snapshot` wiring immediately below the diffing block.
- `strategy/kairos_mtm.py`: `OpenPositionView`, `DailySnapshot`, `compute_daily_snapshot`.
- `docs/tickets/APPENDIX-A-standards.md` for style/testing/commit conventions.
- Related, likely-fixed-as-a-consequence: `docs/tickets/BUG-02-admission-check-defeated-by-same-day-trades.md`.
