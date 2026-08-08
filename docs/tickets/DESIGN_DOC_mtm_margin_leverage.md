# Kairos Paper Trading: Mark-to-Market Equity, Margin Model and Leverage Support

**Version:** 1.0
**Date:** 2026-08-07
**Target:** Kimi Code implementation
**Scope:** `strategy/kairos_papertrade.py` + one new module; no changes to signal generation, strategies, or the discovery pipeline
**Prerequisite reading:** `docs/papertrade_loss_analysis.md` (Factor 1 accounting bugs, Factor 2 exposure), `docs/papertrade_tickets/02-portfolio-exposure-cap.md`

---

## 1. Motivation

The current paper-trade layer cannot answer the question "what happens if we use
IBKR leverage":

1. **No mark-to-market.** `build_closed_trade_equity_curve()` in
   `kairos_papertrade.py` builds equity as a step function over *closed* trades
   only. Its own docstring states it does not capture intra-trade unrealized
   swings, so `pct_max_drawdown` and `sharpe` are understated relative to a true
   continuous series. Margin calls and forced liquidations happen on
   *unrealized* losses; a closed-trade curve structurally cannot represent them.
2. **No margin accounting.** Every position is currently sized against
   `account.cash` (`alloc_eur = row["alloc"] / 100.0 * cash`), i.e. a spot-only,
   fully-funded model. There is no concept of initial margin, maintenance
   margin, free margin, or financing cost. `AllocationConfig.gross_cap_pct=100`
   hard-codes "no leverage" one layer up.
3. **No carry costs.** Leveraged exposure held overnight is not free. The
   measured 0.15% round-trip cost (spread/slippage/fx) matches the gating
   assumption, but financing on CFD notional and borrow fees on shorts are not
   modeled at all. Over a 6-month window at >1x gross this is a first-order
   term.

This document specifies a daily mark-to-market (MTM) equity curve, a
config-driven margin model with liquidation, financing-cost accrual, and the
CLI/allocation knobs to run leveraged backtests. It deliberately *includes* a
portfolio-level exposure cap, subsuming ticket 02 (a margin model is a stricter,
economically grounded version of the same idea).

Non-goals: real broker connectivity (roadmap Phase 5), portfolio margin
modeling, intra-bar liquidation (we evaluate margin once per bar close), and any
change to how signals are produced.

---

## 2. Current state (what we build on)

| Existing piece | Location | Relevance |
|---|---|---|
| Closed-trade equity curve | `kairos_papertrade.py::build_closed_trade_equity_curve` | Replaced as the primary curve; kept for comparison output |
| Corrected per-trade PnL | `kairos_papertrade.py::compute_corrected_realized_pnl` | Philosophy reused: phantom's stored fields + Kairos-side correction |
| Cash reconciliation warning | `kairos_papertrade.py::_reconcile_cash_and_log` | Extended to MTM numbers |
| Window-end position removal | `kairos_papertrade.py::remove_all_open_positions` | Pattern for direct-DB cash/position manipulation reused by liquidation |
| Instrument typing | `kairos_papertrade.py::map_instrument_type`, `_CFD_TICKER_RE` | Replaced by a richer asset-class classifier (section 4.1) |
| Day loop | `kairos_papertrade.py::main` (report replay + `runner.backtest` per day) | Hooks for MTM snapshot, financing accrual, liquidation check |
| Allocation caps | `strategy/allocation.py::AllocationConfig` (`gross_cap_pct`, `max_pos_pct`, `max_cluster_pct`, `kelly_mult`) | New leverage fields added; defaults unchanged |
| phantom margin plumbing | `phantom` account API exposes `get_margin_summary` | Not used as source of truth (phantom cash is untrustworthy for shorts); margin is computed Kairos-side |
| phantom position status enum | `positions.status` CHECK allows `open`, `closed`, `liquidated` | Liquidation uses the existing `liquidated` status, no schema change |

Confirmed phantom bugs that constrain the design (from
`docs/papertrade_loss_analysis.md` Factor 1): phantom's cash tracking is
direction-blind for short positions, and `realized_pnl` omits
`fx_conversion_cost`. **Consequence: all margin and MTM math is computed
Kairos-side from position rows + price bars. phantom is used for order
fill/SL/TP mechanics only; its cash and equity numbers are never inputs to
margin decisions.**

---

## 3. Regulatory reference values (defaults for the IBKR retail config)

These ship as data, not code, and are editable without touching logic. Values
below are the ESMA retail product-intervention caps as applied by EU brokers
including IBKR entities, plus IBKR-typical financing spreads. They are defaults
to be verified against the current IBKR schedule before any real-money use.

| Asset class | Max leverage | Initial margin | Maintenance margin (model default) | Financing (long) |
|---|---|---|---|---|
| FX majors | 30:1 | 3.33% | 1.67% (50% of initial) | benchmark + 1.5% |
| FX minors, gold, major indices | 20:1 | 5% | 2.5% | benchmark + 1.5% |
| Other indices, other commodities | 10:1 | 10% | 5% | benchmark + 1.5% |
| Individual stocks (CFD) | 5:1 | 20% | 10% | benchmark + 1.5% |
| Crypto (CFD, where offered) | 2:1 | 50% | 25% | benchmark + 2.5% (verify) |
| Crypto spot (IBKR Ireland, via Zero Hash) | 1:1 | 100% | n/a (no margin) | none |
| US stocks (Reg-T margin account) | 2:1 overnight | 50% | 25% long / 30% short | BM + 1.5% tiered (verify) |

Additional modeled rules:

- **50% close-out rule:** when account equity falls to 50% of the aggregate
  initial margin requirement on open CFD positions, the broker must begin
  closing positions. This is the liquidation trigger (section 4.4).
- **Negative balance protection:** retail CFD account equity floored at zero.
  The model clamps equity at 0 and halts new orders; it does not simulate
  debt.
- **House margins:** IBKR applies stricter margins on concentrated/volatile
  names. The config supports per-symbol overrides for exactly this.
- Day-count convention for financing: ACT/360, charged daily on the day's
  closing notional.

Sources (retrieve current values at implementation time):
- ESMA CFD product intervention measures (leverage caps, 50% close-out,
  negative balance protection)
- interactivebrokers.eu margin and CFD financing pages
- IBKR Ireland crypto announcement (spot, unleveraged, March 2026)

---

## 4. Design

### 4.1 Asset-class classification and margin config

New file: `config/margin_ibkr.yaml` (loaded once per run; path overridable via
`--margin-config`).

```yaml
# config/margin_ibkr.yaml
base_currency: EUR
benchmark_annual_pct: 3.15          # configurable; verify current EUR benchmark
negative_balance_protection: true
closeout_fraction: 0.5              # ESMA: liquidate at 50% of initial margin
classes:
  fx_major:
    symbols: ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X"]
    initial_margin_pct: 3.33
    maintenance_margin_pct: 1.67
    financing_spread_pct: 1.5
  fx_minor:
    match: "=X$"                    # regex fallback for any other FX pair
    initial_margin_pct: 5.0
    maintenance_margin_pct: 2.5
    financing_spread_pct: 1.5
  index_gold_major:
    symbols: ["GC=F", "^GSPC", "^IXIC", "^DJI", "SPY", "QQQ"]
    initial_margin_pct: 5.0
    maintenance_margin_pct: 2.5
    financing_spread_pct: 1.5
  commodity_other:
    match: "=F$"                    # other futures proxies after index_gold_major
    initial_margin_pct: 10.0
    maintenance_margin_pct: 5.0
    financing_spread_pct: 1.5
  crypto_cfd:
    enabled: false                  # IBKR EU retail availability uncertain; off by default
    initial_margin_pct: 50.0
    maintenance_margin_pct: 25.0
    financing_spread_pct: 2.5
  crypto_spot:
    match: "-USD$"
    initial_margin_pct: 100.0       # unleveraged: full notional locked
    maintenance_margin_pct: 0.0
    financing_spread_pct: 0.0
  equity_cfd:
    match: null                     # default for anything unmatched (plain tickers)
    initial_margin_pct: 20.0
    maintenance_margin_pct: 10.0
    financing_spread_pct: 1.5
overrides:                          # per-symbol house-margin overrides
  # "LDO-USD": {initial_margin_pct: 100.0}
short_borrow_annual_pct:
  default: 1.0                      # placeholder; real borrow fees are per-name
  overrides: {}
```

Classification function (new, pure, unit-testable):

```python
# strategy/kairos_margin.py
def classify_symbol(symbol: str, cfg: MarginConfig) -> MarginClass:
    """First explicit `symbols` membership in config order, then first
    matching `match` regex in config order, then the class with match: null.
    Crypto special case: if crypto_cfd.enabled is false, '-USD' tickers fall
    through to crypto_spot automatically."""
```

Note this replaces `_CFD_TICKER_RE`-based typing for margin purposes only;
`map_instrument_type()` stays as-is for phantom order placement.

### 4.2 Daily mark-to-market equity curve

New module `strategy/kairos_mtm.py` (pure functions over plain data; no phantom
import, no GPU, no network):

```python
@dataclass
class OpenPositionView:
    ticker: str
    direction: str          # "long" | "short"
    entry_price: float
    quantity: float
    entry_costs: float      # commission_entry + spread_cost + slippage_cost + fx_conversion_cost

@dataclass
class DailySnapshot:
    date: datetime
    cash: float
    unrealized_pnl: float
    equity: float           # cash + unrealized_pnl (post-financing cash)
    gross_notional: float
    initial_margin_used: float
    maintenance_margin_used: float
    free_margin: float      # equity - initial_margin_used
    margin_utilization: float  # initial_margin_used / equity, 0 if equity <= 0
    financing_accrued_day: float
    liquidations: int

def unrealized_pnl(pos: OpenPositionView, close_price: float) -> float:
    """Direction-aware, computed Kairos-side (never phantom cash):
    long:  (close - entry) * qty
    short: (entry - close) * qty"""

def compute_daily_snapshot(positions, bars_by_ticker, cash, cfg) -> DailySnapshot: ...
```

The corrected cash itself comes from the same reconciliation philosophy already
in place: start-of-window capital plus cumulative corrected realized PnL
(`compute_corrected_realized_pnl`) of positions closed so far, minus cumulative
financing accrued, plus refunds from `remove_all_open_positions` semantics.
Concretely: maintain a `corrected_cash` running value in the day loop, updated
on every fill (entry costs + locked-notional treatment per section 4.3) and
every close (direction-aware cash effect + fx correction), and cross-checked
against phantom's raw cash via the existing `_reconcile_cash_and_log` pattern
(warning-only, expected to diverge when shorts exist).

**Persistence:** new SQLite table in `data/phantom_ledger/phantom.db` (sidecar
table, Kairos-owned, no phantom schema change):

```sql
CREATE TABLE IF NOT EXISTS kairos_mtm_daily (
    account_name TEXT NOT NULL,
    date TEXT NOT NULL,               -- ISO date of the bar close
    cash REAL, unrealized_pnl REAL, equity REAL,
    gross_notional REAL,
    initial_margin_used REAL, maintenance_margin_used REAL,
    free_margin REAL, margin_utilization REAL,
    financing_accrued_day REAL, financing_accrued_total REAL,
    liquidations INTEGER,
    PRIMARY KEY (account_name, date)
);
```

**Day-loop integration:** after each day's `runner.backtest(...)` call (which
fills orders and evaluates SL/TP against that day's bars), fetch each open
position's closing price from the same `_IntradayFallbackProvider` ladder,
compute the snapshot, insert the row. Bars are already fetched for the
backtest; cache them per-day to avoid double-fetching.

### 4.3 Margin accounting at order admission

In the day loop, before placing each day's selected orders:

```python
def admission_check(order_notional: float, ticker: str, account: DailySnapshot,
                    cfg: MarginConfig) -> bool:
    """Reject (skip) a new order if post-trade state would breach limits:
    - initial_margin_used_post <= equity * cfg.margin_utilization_cap
    - equity_post > 0
    Returns False = skip order, log line, count as MARGIN_REJECTED."""
```

Margin treatment per class:

- `initial_margin_pct < 100`: position locks `notional * im_pct` of margin;
  cash moves only by entry costs (CFD/margin semantics).
- `initial_margin_pct == 100` (crypto spot): full notional is debited from
  corrected cash at entry (current behavior), no margin usage, no financing.

New `AllocationConfig` fields (defaults preserve current behavior exactly):

```python
max_leverage: float = 1.0            # >1 enables margin mode
margin_utilization_cap: float = 0.8  # fraction of equity usable as initial margin
gross_cap_pct: float = 100           # may be raised (e.g. 150/200) when leveraged
```

When `max_leverage == 1.0` the admission check is a no-op and cash handling is
byte-identical to today; the whole feature is opt-in.

### 4.4 Liquidation model

Evaluated once per day at bar close, after the MTM snapshot:

```python
def liquidation_check(snapshot: DailySnapshot, positions, cfg) -> list[str]:
    """ESMA close-out: if equity < closeout_fraction * initial_margin_used,
    liquidate positions (largest maintenance-margin release first) until
    equity_post >= closeout_fraction * initial_margin_used_post or no
    positions remain. Returns tickers liquidated."""
```

Liquidation execution (mirrors `remove_all_open_positions`'s direct-DB
pattern):

1. Close the position at that day's close price by writing phantom rows with
   `status='liquidated'` (the enum already allows it), nulling
   `orders.position_id` first (FK is RESTRICT).
2. Apply the *corrected* cash effect Kairos-side (direction-aware gross PnL
   minus exit costs minus fx), since phantom's own close-path cash is wrong
   for shorts. Never call phantom's `PositionAPI.close()` for liquidations.
3. Log one Telegram line per liquidation event via the existing `_notify`.
4. Negative balance protection: clamp equity at 0, set a run-level
   `ruined=True` flag, stop opening new positions for the rest of the window
   (existing positions still resolve via SL/TP).

Deterministic ordering (largest maintenance release first) keeps the rule
testable; document that real IBKR liquidation order differs.

### 4.5 Financing and borrow-cost accrual

Per open position, per day, at bar close:

```python
def daily_financing(pos: OpenPositionView, close_price: float, cls: MarginClass,
                    benchmark_annual_pct: float) -> float:
    """Long CFD/margin: notional_close * (benchmark + spread) / 360.
    Short: notional_close * (benchmark - spread) / 360 CREDITED if positive,
    plus borrow fee notional_close * borrow_pct / 360 always debited.
    Spot (im == 100): 0."""
```

- Deduct from corrected cash daily; accumulate per-position and per-run totals.
- Persist daily total in `kairos_mtm_daily.financing_accrued_day/_total`; the
  run-level total lands in the JSON metrics as `financing_total_eur`.
- Financing is charged on positions open at that day's close; entry day counts,
  exit day does not (document the convention).

### 4.6 Metrics and reporting changes

`compute_final_metrics()` gains a parallel computation over
`kairos_mtm_daily`:

```python
{
  # existing keys unchanged (closed-trade curve), plus:
  "mtm_total_return_pct": ...,
  "mtm_max_drawdown_pct": ...,     # from daily MTM equity, the honest number
  "mtm_sharpe": ...,
  "mtm_margin_utilization_peak": ...,
  "mtm_financing_total_eur": ...,
  "mtm_liquidation_events": ...,
  "mtm_ruined": false,
}
```

Both curve families are reported side by side so the curve-shape tradeoff
documented in the loss analysis (closed-trade curve understates DD) becomes a
measured quantity instead of an assumption. The HTML report gains a second
panel: MTM equity with drawdown shading, margin utilization line, and
liquidation markers.

### 4.7 CLI additions (`kairos_papertrade.py`)

```
--margin-config PATH     default config/margin_ibkr.yaml
--max-leverage FLOAT     default 1.0 (off)
--margin-utilization FLOAT  default 0.8
```

All three forward into `AllocationConfig`/the day loop. No change to report
generation, prewarm, or the prediction cache.

---

## 5. Engine modifications (exact touch points)

1. `strategy/kairos_margin.py` (new): `MarginConfig` loader, `MarginClass`,
   `classify_symbol`, margin/financing math.
2. `strategy/kairos_mtm.py` (new): `OpenPositionView`, `DailySnapshot`,
   `unrealized_pnl`, `compute_daily_snapshot`, `admission_check`,
   `liquidation_check`, `daily_financing`. All pure.
3. `strategy/kairos_papertrade.py`:
   - `main()`: load margin config; maintain `corrected_cash`; after each
     `runner.backtest` call, snapshot MTM -> `kairos_mtm_daily`, accrue
     financing, run liquidation check.
   - Order placement block: gate each order through `admission_check`; CFD
     orders stop debiting full notional (locked-margin semantics).
   - `compute_final_metrics()`: add MTM metric block (4.6).
   - `write_html_report()`: second panel (4.6).
   - `_reconcile_cash_and_log()`: also reconcile corrected cash + open
     unrealized vs phantom raw equity where meaningful.
4. `strategy/allocation.py`: three new `AllocationConfig` fields (4.3).
5. `config/margin_ibkr.yaml`: new (4.1).
6. `docs/papertrade_tickets/02-portfolio-exposure-cap.md`: mark subsumed on
   completion (admission check + utilization cap is the exposure cap).

---

## 6. Testing plan

All unit tests live in `tests/unit/`, no GPU/network/phantom install required
for the pure modules (same property as the existing suite).

### 6.1 Unit tests — `kairos_margin` / `kairos_mtm`

| Test | Setup | Pass criteria |
|---|---|---|
| classify majors FX | `EURUSD=X` | fx_major, im 3.33% |
| classify fallback order | `CL=F`, `GC=F`, `BTC-USD`, `AAPL` | commodity_other / index_gold_major / crypto_spot / equity_cfd |
| classify crypto_cfd disabled | `crypto_cfd.enabled=false`, `BTC-USD` | crypto_spot, im 100% |
| symbol override | overrides entry for one ticker | override wins over regex |
| unrealized long/short | qty 10, entry 100, close 105 | +50 / -50 |
| snapshot margin math | 2 positions, known notionals | im/mm/free_margin/utilization exact |
| admission accept | utilization below cap post-trade | True |
| admission reject | order pushes utilization over cap | False, notional untouched |
| financing long | 10000 notional, bm 3% + 1.5%, 1 day | 10000 * 0.045/360 |
| financing short with borrow | short, bm 3% - 1.5%, borrow 1% | credit 0.015/360, debit 0.01/360 |
| financing spot | im == 100% class | 0 |
| liquidation trigger | equity = 0.49 * im_used | liquidation list non-empty |
| liquidation ordering | 3 positions different mm | largest mm release first |
| liquidation sufficiency | post-liquidation equity/im restored | loop terminates, invariant holds |
| negative balance clamp | equity would go negative | clamped at 0, ruined=True |
| leverage off | max_leverage=1.0 | admission no-op, cash path identical to legacy |

### 6.2 Repro test against the frozen fixture

`tests/unit/test_kairos_papertrade_loss_repro.py` pins the 2026-07-23 run.
Add a sibling test that replays the same frozen fixture through the MTM path:

- `kairos_mtm_daily` row count equals trading days in the window.
- Final MTM equity equals final corrected closed-trade equity (all positions
  closed or removed at window end, so the curves must converge at the end
  point; this is a strong invariant).
- MTM max drawdown >= closed-trade max drawdown (the documented
  understatement), report both.

### 6.3 Integration smoke test

One 2-week `--months-back 0.5` run with `--max-leverage 2.0` on a small asset
set, verifying: no crash, `kairos_mtm_daily` populated, JSON contains the MTM
block, and `mtm_max_drawdown_pct >= pct_max_drawdown`.

---

## 7. Implementation order

**Phase 1 — pure math, no engine changes (1 subagent-day):**
1. `kairos_margin.py` + `config/margin_ibkr.yaml` + classification tests
2. `kairos_mtm.py` snapshot/financing/admission/liquidation + tests

**Phase 2 — engine wiring (1-2 subagent-days):**
3. `kairos_papertrade.py` day-loop hooks: corrected cash, MTM snapshot
   persistence, financing accrual
4. Admission check + leveraged order semantics
5. Liquidation execution path + Telegram notification
6. Metrics + HTML reporting

**Phase 3 — validation (orchestrator):**
7. Frozen-fixture repro test (6.2)
8. Leverage-off regression: a `--max-leverage 1.0` run must reproduce current
   JSON metrics bit-for-bit on a short window
9. Smoke test (6.3), then a full 6-month `--max-leverage 2.0` comparison run

---

## 8. Acceptance criteria summary

- [ ] MTM daily equity curve persisted for every replayed day
- [ ] Final MTM equity == final closed-trade equity on the frozen fixture
- [ ] MTM max drawdown >= closed-trade max drawdown on every run
- [ ] Margin admission check rejects orders that would breach utilization cap
- [ ] Liquidation triggers at 50% of initial margin, deterministic order,
      `liquidated` status used, corrected cash applied Kairos-side
- [ ] Financing/borrow accrues daily, appears in JSON metrics
- [ ] `--max-leverage 1.0` reproduces legacy behavior exactly
- [ ] All new unit tests pass with no GPU/network/phantom dependency
- [ ] `ROADMAP.md` / `strategy/PIPELINE.md` untouched; README papertrade
      section updated with the new flags

---

## 9. Out of scope (explicit)

- Real IBKR API connectivity, live margin data, or execution (roadmap Phase 5).
- Intra-bar margin evaluation (bar-close granularity only).
- IBKR Portfolio Margin (>110k USD accounts) — Reg-T/ESMA rules only.
- Cross-currency margin haircut beyond the existing fx cost model.
- Monte Carlo projection and the leverage sweep study — those are the
  *consumers* of this work, specced separately.
