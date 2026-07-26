# Kairos paper-trade loss analysis (run 2026-07-23)

## 0. Update (2026-07-26): fixes applied, and what a fresh rerun actually showed

Three fixes from this analysis were implemented and verified:

1. **`--base-only` now defaults to `False`** (finetuned overlay used by default).
2. **Window-end open positions are now removed, not force-closed**
   (`remove_all_open_positions` replaces `force_close_all_open`): refunded and
   excluded from every statistic instead of manufacturing a same-day "manual" exit.
3. **The cash-reconciliation gap is root-caused and fixed at Kairos's metrics layer**
   — see the rewritten Factor 1 section below for the full derivation. Short version:
   two confirmed `phantom_ledger` bugs (a direction-blind cash bug for short
   positions, and `fx_conversion_cost` omitted from `realized_pnl`) fully explained
   the €48.71 gap. `compute_final_metrics` now builds its own corrected closed-trade
   equity curve instead of trusting phantom's own (buggy-for-shorts) cash tracking.

**Recomputing the ORIGINAL 2026-07-23 run's 539 trades with the fixed accounting**
reveals it was actually **profitable: +€30.65 (+15.33%)**, not the reported -€8.40
loss — the paradox described in Section 1 below was itself a symptom of the
accounting bugs, not a real phenomenon to explain (see the rewritten Factor 1).

**A genuinely fresh rerun** (2026-07-26, same command, all three fixes active,
`base_only=false`) over a new but overlapping 6-month window produced
**total_profit_eur = -€16.47 (-8.23%)**, `pct_profit_per_trade = -0.26%`, 423 trades,
max drawdown 9.21%, Sharpe -1.51 — a real loss on genuinely different trades, not a
reopening of the original bug (per-trade mean and total return now agree in sign:
both negative). This is **not** a controlled apples-to-apples comparison against the
original run: the window shifted a few days forward, the finetuned overlay barely
engaged (it fired on only 2 of ~183 daily reports — the win/loss shift is not
attributable to the default flip), and day-to-day variance in signal generation
(updated DB state, viability rankings, etc.) is expected between any two runs of a
live system. If a true before/after comparison is wanted, rerun with
`--effective_per` pinned to the *original* window's end (`20260723 1458`) so both
runs replay the identical historical window.

`tests/unit/test_kairos_papertrade_loss_repro.py` now pins both: the retroactively-corrected
2026-07-23 numbers (proving the fix) and the fresh 2026-07-26 run's numbers (a
current snapshot, not a target).

---

**Source run:** `strategy/kairos_papertrade.py --months-back 6 --interval 1d --html`,
effective window 2026-01-22 → 2026-07-23, `1d` bars, €200 starting capital, broker
`IBKR`, `--base-only` (the default at time of run), `top_n=3`.

**Recorded output:**
`results/kairos_signals_papertrade_202607231458_202601221458_1d_6.0m.json`. The exact
Phantom Ledger state from that run is preserved at `data/phantom_ledger/phantom.db`
and frozen as a test fixture at
`tests/data/kairos_papertrade_20260723_phantom.db`, reproduced deterministically by
`tests/unit/test_kairos_papertrade_loss_repro.py`.

All numbers in this document come from that recorded run (verified by re-running the
production `compute_final_metrics()` against the frozen DB, and by direct queries
against it), not from a fresh live rerun.

## 1. Summary

| Metric | Value |
|---|---|
| Total profit | **-€8.40** |
| Total return | **-4.01%** |
| Mean return per trade | **+0.57%** |
| Max drawdown | 42.56% |
| Sharpe | -0.317 |
| Trades | 539 (over 183 trading days, ~2.9/day) |

**The central paradox:** the average trade made money (+0.57%), but the account lost
money overall (-4.01%). This is not a rounding artifact — of the 539 closed trades,
131 (24.3%) were winners averaging +€0.93, and 408 (75.7%) were losers averaging
-€0.20. That's a **low win rate with a ~4.6x payoff ratio** (classic
trend/optionality-style edge), and summed alone those trades net **+€40.32**. Yet the
account's actual cash fell from €200.00 to €191.60 (-€8.40). Two separate mechanisms
explain the gap, both detailed below:

1. **A real, unreconciled accounting gap of ~€48.7** between the sum of each trade's
   own recorded `realized_pnl` and the account's actual net cash change — a fact this
   analysis surfaces but does not fully root-cause (see Factor 1).
2. **Uncapped concurrent exposure**: `top_n=3` only limits *new* positions opened per
   day; nothing caps *total open* notional. Account cash was observed falling to
   **€74.22 (37% of capital remaining)** at one point, and cumulative traded notional
   over the window was **€9,665 — 48x capital turnover**. Stacking many concurrent,
   correlated crypto/CFD bets against a thin per-trade edge is exactly the recipe for
   large drawdowns eating a positive arithmetic mean (variance/volatility drag) — see
   Factor 2.

Every position in this run was typed `"cfd"` (100% — the traded universe is crypto/
futures/forex tickers), average position size was **€17.93 (~10-15% of capital)**, and
the realized average round-trip cost was **exactly 0.15% of notional** — matching
`strategy/allocation.py`'s `round_trip_cost_pct=0.15` assumption almost exactly. This
run also used **`base_only=true`** (the CLI default at the time of this run), meaning it never used the
finetuned-model overlay, even though 3 finetuned models are already `accepted` in
`data/pipeline_results.db` for asset groups that overlap with this run's traded/losing
tickers. *Note: The default has since been flipped to include the finetuned overlay by
default; see Factor 6 below for recommendations on re-running with the new default.*

## 2. Contributing factors (identification)

Walking the money from raw model prediction through to the final equity number, these
are the levers, in Kairos's control unless noted otherwise:

- **Equity/PnL accounting** (`kairos_papertrade.py::compute_final_metrics`) — how the
  final numbers are computed from the simulation's recorded state. *In Kairos's
  control* for its own glue code; *partly external* for the underlying cash/PnL
  bookkeeping inside the `phantom` package.
- **Portfolio-level exposure** — nothing in `kairos_papertrade.py`'s day-loop caps
  total concurrent open notional. *In Kairos's control.*
- **Transaction costs** — spread/slippage/fx/commission model, and the
  `round_trip_cost_pct` assumption used to gate signals. *Cost model is external*
  (`phantom`'s broker-profile JSON + `CostEngine`); *the assumption used for gating is
  in Kairos's control* (`strategy/allocation.py`).
- **Order execution mechanics** — one-report lag, market-order fill price, stop/target
  anchoring, same-bar SL/TP tie resolution. *Mostly in Kairos's control*
  (`kairos_papertrade.py`'s order construction); *tie-resolution mode is a `phantom`
  default Kairos doesn't currently override*.
- **Position sizing / allocation** (`strategy/allocation.py`) — Kelly shrinkage,
  per-position/cluster/gross caps, the `NEG_EV_NET` gate. *Fully in Kairos's control.*
- **Model selection (base vs. finetuned)** — `kairos_papertrade.py` defaults to
  finetuned-overlay mode when an accepted model exists, or via `--base-only` to opt
  back to the base model; mediated by the `finetuned_models` accept gate (same logic
  as `kairos_signals.py`'s two-pass overlay). *Fully in Kairos's control.*
- **Signal generation & strategy filtering** (`kairos_orchestrator.py`,
  `kairos_meta.py`, `kairos_execution.py`) — entropy/kurtosis/liquidity meta-filters,
  `min_ev_pct`, disabled-strategy lists. *Fully in Kairos's control.*
- **Underlying model/signal fitness** — the base Kronos model's real predictive
  quality per asset, and the oracle (perfect-foresight) ceiling it's compared against.
  *Partly in Kairos's control* (finetuning is a lever; the base model's raw capability
  is a harder ceiling).

## 3. Factors, in process order (last → first)

Ordered starting from the stage closest to the observed loss number, working backward
toward the root of the pipeline:

1. **Equity/PnL accounting & reporting**
2. **Portfolio-level risk aggregation / concurrent exposure**
3. **Transaction cost accrual**
4. **Order execution mechanics**
5. **Position sizing / capital allocation**
6. **Model selection: base vs. finetuned overlay**
7. **Signal generation & strategy filtering**
8. **Underlying model/signal fitness**

## 4. Per-factor: statistics to optimize + concrete levers

### 1. Equity/PnL accounting & reporting

**RESOLVED — root cause found, fixed in Kairos's metrics layer.** The +€48.71
reconciliation gap (`capital + Σrealized_pnl - final_cash`) flagged below was not a
mystery left open by this document; it has since been root-caused to two confirmed,
precisely-located bugs inside the external `phantom_ledger` package itself, reproduced
against individual rows of the frozen fixture DB
(`tests/data/kairos_papertrade_20260723_phantom.db`), not just aggregates:

1. **Direction-blind cash debit/credit for short positions (~€39.05 of the gap, the
   dominant driver).** Both the entry-side cash debit
   (`phantom/engine/order_manager.py`, `OrderManager.handle_fill`, line 300:
   `total_deduction = order.fill_price * order.quantity + costs.total`) and every
   exit-side cash credit (`phantom/engine/simulation_engine.py`,
   `SimulationEngine.run_backtest`, line 214:
   `cash_return = exit_price * position.quantity - exit_costs.total`; and
   `phantom/api/positions.py`, `PositionAPI.close`, lines 93 and 128, same pattern)
   use raw `price * quantity` with **no check of `position.direction` at all**. For a
   `"long"` position this is correct (net cash effect nets to
   `(exit-entry)*quantity - costs`, matching the trade's real gross P&L). For a
   `"short"` position it is backwards: cash still moves by `(exit-entry)*quantity`,
   the *opposite* sign of a short's real gross P&L of `(entry-exit)*quantity` — so a
   **winning short trade decreases** phantom's tracked cash and a losing short
   increases it — even though the position's own stored `realized_pnl`
   (`phantom/engine/position_manager.py`, `PositionManager.close`, lines 314-317) is
   computed correctly, direction-aware. This run traded 130 short positions (of 539);
   `2 × Σ(gross_pnl over those 130 shorts)` = **€39.05**, exactly the dominant term of
   the gap.
2. **`fx_conversion_cost` omitted from `realized_pnl` (~€9.67 of the gap, all
   positions).** `fx_conversion_cost` is charged to `account.cash` at entry
   (`order_manager.py:300`'s `costs.total` includes it), but
   `position_manager.py`'s `close()` (lines 319-327) computes `realized_pnl` from
   `all_costs = commission + spread + slippage` only — it never subtracts
   `fx_conversion_cost` back out. Every position in this EUR-base-currency run paid
   this cost (`fx_required = base_currency != "USD"` is `True` for EUR), so it affects
   all 539 trades, not just the shorts.

Together these exactly reconcile the gap: `2 × Σgross_pnl(shorts) + Σfx_conversion_cost`
= €39.05 + €9.67 = **€48.71**, matching the observed gap to 5 significant figures
(verified via a per-position, per-timestamp replay of the entire 6-month `equity_curve`
against phantom's own stored cost/price fields — not just an aggregate coincidence).

**Consequence — the reported -€8.40 loss on this run is itself an artifact of bug #1,
not a real trading outcome.** `total_profit_eur`/`pct_profit`/`sharpe`/
`pct_max_drawdown` were being derived from `accounts.get_aggregate_equity()`, i.e.
phantom's own buggy short-position cash tracking. Recomputing correctly (capital +
Σ(direction-and-fx-corrected per-trade P&L) over the same 539 historical trades) gives
**+€30.65 (+15.33%)** on this exact historical run — a profit, not a loss. This does
**not** mean the strategy is confirmed profitable going forward (it's one 6-month
window, and Factor 2's uncapped-concurrent-exposure critique below still stands on its
own merits) — but it does mean the original "positive per-trade mean, yet the account
lost money" paradox in section 1 above was never a real phenomenon to explain; it was
this accounting bug.

**Fix applied in `kairos_papertrade.py`** (the bug lives in the external `phantom`
package, which is git-pinned via `uv.lock` and can't be durably patched in
`.venv/site-packages`, so the fix is a workaround at Kairos's metrics layer, not a
patch to `phantom` itself):
- `compute_corrected_realized_pnl()` — the true per-trade P&L, correcting bug #2
  (subtracts `fx_conversion_cost`; `realized_pnl`'s own gross-P&L *direction* was
  already correct and needs no change). Used by `compute_pct_profit_per_trade`.
- `build_closed_trade_equity_curve()` — replaces `accounts.get_aggregate_equity()` as
  the input to `calculate_metrics()` (drives `total_profit_eur`/`pct_profit`/`sharpe`/
  `pct_max_drawdown`). It reconstructs equity purely from each **closed** position's
  own corrected P&L (a step function at each trade's exit, prefixed with a `capital`
  point at the window's start), sidestepping bug #1 entirely rather than trying to
  patch phantom's per-bar series. Tradeoff: this is a closed-trade curve, not a true
  continuous mark-to-market series — it does not capture intra-trade unrealized
  swings from positions that are still open at some intermediate point in time, so
  `pct_max_drawdown`/`sharpe` computed from it are **understated** relative to a true
  continuous series (this run's drawdown falls from the previously-reported 42.56% to
  6.39% under the corrected curve — some of that drop is the real accounting fix, but
  some is this curve-shape tradeoff, and the two aren't currently separated).
- `compute_final_metrics()` now also calls `_reconcile_cash_and_log()`, which compares
  `capital + total_profit_eur` (Kairos's corrected number) against phantom's raw
  `account.cash` and logs a warning (not a hard failure) whenever they diverge by more
  than 1 cent — expected whenever the run holds short positions (bug #1), but a
  free early-warning if a long-only run ever shows a nonzero gap (a *new*,
  uninvestigated divergence) or if an eventual `phantom` upgrade fixes bug #1 (the gap
  should then shrink toward zero on runs with shorts too).

**Remaining open question, deliberately not resolved here:** whether
`pct_profit_per_trade`'s definition (unweighted mean) is the right summary statistic
to report alongside `pct_profit` (size-weighted/geometric) — the two can still point
in different directions in general, independent of the accounting bugs above.

### 2. Portfolio-level risk aggregation / concurrent exposure

The day-loop (`kairos_papertrade.py:504-553`) sizes each day's *new* orders against
`cash = client.accounts.get(account_id).cash` (line 508) with `top_k=args.top_n` (default
3) and `max_pos_pct=15` — but nothing caps how many *previously opened, still-open*
positions can be outstanding at once. Measured result: account cash fell as low as
**€74.22 (37% of capital)** — far more simultaneous capital-at-risk than "3 new
positions/day at ≤15% each" suggests on its own, because positions accumulate across
days until they individually hit SL/TP.

**4.1 Statistic to optimize:** peak simultaneous capital-at-risk (%), and its direct
consequence, **max drawdown** (currently 42.56%). This is the single biggest lever on
why a positive per-trade mean (+0.57%) turned into a negative total return
(-4.01%) — high concurrent exposure amplifies variance, and geometric/compounded
returns suffer disproportionately from variance even when the arithmetic edge is
positive ("volatility drag").

**4.2 Concrete changes:**
- Add an explicit portfolio-level exposure cap in the day-loop — e.g. skip opening new
  positions once total open notional exceeds some fraction of equity, independent of
  the *daily* `gross_cap_pct=100` (which only constrains that day's *new* batch, not
  the running total across all still-open positions from prior days).
- Consider capping total *concurrent* position count directly (not just new-per-day),
  or shrinking `max_pos_pct` as a function of currently-committed capital rather than
  a flat 15%.
- `tests/unit/test_kairos_papertrade_loss_repro.py::TestConcurrentExposureAndTurnover`
  pins the current €74.22 cash floor and €9,665 cumulative notional (48x turnover) so
  a fix here is directly measurable.

### 3. Transaction cost accrual

Measured realized round-trip cost across all 539 trades: **exactly 0.15% of
notional** (spread €2.90 + slippage €1.93 + fx €9.67 + commission €0.00, over €9,665
total notional) — matching `strategy/allocation.py`'s `round_trip_cost_pct=0.15`
assumption almost exactly. **This rules out "the cost assumption is too optimistic" as
the main driver of the loss** — it's empirically accurate on average. Two real issues
remain: commission is silently always €0.00 (a schema mismatch between `phantom`'s
`"tiered"` commission model and the bundled IBKR broker-profile JSON's tier keys,
external to Kairos), and 0.15% is a single constant applied uniformly even though the
top losing tickers (`LDO-USD`, `AAVE-USD`, `ATOM-USD`, `XTZ-USD`, `AXS-USD`) are
lower-liquidity alts likely to have wider real-world spreads than majors.

**4.1 Statistic to optimize:** per-asset-class realized cost % (not just the blended
average), and whether `min_ev_pct`/`round_trip_cost_pct` should vary by
liquidity/asset-class rather than being one global constant.

**4.2 Concrete changes:**
- Fix the IBKR broker-profile commission schema mismatch (the `"tiered"` branch reads
  `tier.get("up_to")`/`tier.get("rate")`, keys that don't exist in the bundled
  `ibkr.json`) so simulated commission isn't silently zero — this makes the backtest
  *more* conservative, not less, so fixing it won't explain the current loss, but
  leaving it unfixed means any *future* run understates real trading costs.
  This lives in the external `phantom` package, not this repo.
  - Note: for a genuine IBKR account, commissions matter a lot more for the account's
    tiny average trade size (~€18) than the 0.15% spread/slippage/fx does — worth
    validating against real IBKR crypto/CFD commission schedules before trusting this
    number is close to real-world.
- Consider a per-asset-class (or per-ticker-liquidity) `round_trip_cost_pct` in
  `AllocationConfig` instead of one flat 0.15%, so `NEG_EV_NET` gating is stricter for
  the illiquid alts that dominate the loss list.

### 4. Order execution mechanics

Every order fills at the next real bar's **Open** price (`phantom`'s
`OrderManager.evaluate`), one report-cycle after the signal was generated
(`kairos_papertrade.py`'s "ONE-REPORT LAG" design, module docstring lines 4-9). But
`take_profit`/`stop_loss` on the placed `Order` (lines 527-528) are copied **verbatim**
from the signal's `target`/`stop`, which the originating strategy computed relative to
the **stale, report-time entry price** — not the actual fill price. When a same-bar
tie occurs between hitting stop and target, `phantom`'s default conflict resolution
(`mode="conservative"`, never overridden anywhere in this call chain) always resolves
to **SL wins**. Measured result: of 539 closes, **409 hit stop-loss** vs. only **120
hit take-profit**.

**4.1 Statistic to optimize:** the SL:TP hit-rate ratio (currently ~3.4:1) and the
realized risk/reward ratio actually achieved vs. what each strategy's own
distribution-derived stop/target implied. Also: the size of the overnight gap between
report-time signal `entry` and the actual next-bar-open fill price, which determines
how much the un-rebased stop/target drifts from what was intended.

**4.2 Concrete changes:**
- Re-base `stop`/`target` off the **actual fill price** at order-creation time
  (`kairos_papertrade.py:523-529`) instead of copying the signal-time values verbatim
  — e.g. recompute `stop`/`target` as the same %-distance-from-entry the strategy
  originally intended, applied to the real fill price.
- Evaluate whether `phantom`'s tie-resolution mode should be overridden (it currently
  always favors SL on same-bar hits, structurally suppressing the payoff side of a
  low-win-rate/high-payoff strategy).
- Consider limit orders (bounded slippage on entry) instead of market orders for
  signals where the report-time entry and typical next-bar gap size make market fills
  risky.

### 5. Position sizing / capital allocation

`strategy/allocation.py`'s Kelly-shrinkage sizing (`n0=100`, `min_n=50`,
`kelly_mult=0.35`) combined with `max_pos_pct=15`, `top_k=3` (from `--top-n`), and
`gross_cap_pct=100` determines exactly how big each position is. Given the measured
24.3% win rate / ~4.6x payoff ratio, the *true* Kelly-optimal fraction is thin and
highly sensitive to `p`/`b` estimation error — oversizing here is what turns Factor 2's
uncapped concurrency into large realized drawdown.

**4.1 Statistic to optimize:** the realized Kelly edge (`p_shrunk`, `kelly_frac`) vs.
actual out-of-sample win rate/payoff — and, jointly with Factor 2, the **peak
portfolio-level** (not just per-position) Kelly fraction actually deployed.

**4.2 Concrete changes:**
- Lower `kelly_mult` (currently 0.35) to reduce variance, especially while Factor 2
  (uncapped concurrent exposure) is unaddressed — the two compound.
- Consider sizing at the **portfolio** level (aggregate Kelly across all currently open
  + newly selected positions) rather than per-position-in-isolation, since that's what
  actually drives drawdown.
- Re-validate `min_n=50`/`n0=100` shrinkage constants against how much real sample
  data (`n`) is typically available for the traded tickers — thin-sample signals get
  heavily shrunk toward `p=0.5`, but if `n` is systematically inflated or deflated for
  certain asset classes, sizing will be systematically biased too.

### 6. Model selection: base vs. finetuned overlay

This run was conducted with the old default where `kairos_papertrade.py --base-only`
was **on** (see `kairos_papertrade.py:450-451` at the time of the run), so the entire
6-month backtest used **only the base Kronos model** — confirmed by
`meta.base_only: true` in the recorded JSON. **The default has since been flipped**: as
of the current version, `kairos_papertrade.py` defaults to using the finetuned-overlay
when an accepted model exists, requiring `--base-only` to explicitly opt back to the
base model. Meanwhile `data/pipeline_results.db` already had **3 `accepted`** finetuned
models (covering `ADA-USD/ETH-USD/LINK-USD/SOL-USD`, `AVAX-USD/LINK-USD/SOL-USD/SUI20947-USD`,
and `ADA-USD/DOT-USD/SUI20947-USD/TIA-USD`) that passed the realized-backtest-Sharpe
accept gate in `kairos_pipeline.py`'s
`compare_finetuned_vs_base` — meaning these specific models have *already been shown*
to beat the base model on realized (not oracle) Sharpe for these asset groups.
`SOL-USD` is among this run's traded assets.

**4.1 Statistic to optimize:** per-asset-group realized Sharpe/signal-count,
base vs. finetuned — the exact stat the accept gate already tracks in
`model_results`.

**4.2 Concrete changes:**
- Re-run the same 6-month window with the *new* default (finetuned overlay enabled) to
  see the impact of using the accepted models that have already been vetted as
  outperforming base on realized backtest Sharpe. This now matches what Kairos's *live*
  daily-signals pipeline (`kairos_daily_signals.py`, which runs `kairos_signals.py` with
  the finetuned overlay by default) actually recommends. The old base-only run was
  described above; a finetuned-overlay run would provide the comparison point needed
  to measure the impact of this lever.
- Broaden `kairos_idle_finetune.py`'s finetuning candidate coverage — right now the
  accept-gate pipeline only finetunes asset groups that `select_finetune_candidate`
  ranks, but the idle-GPU trigger script separately only rotates through a hardcoded
  `["BTC-USD","ETH-USD","SOL-USD"]` default list, which doesn't include several of this
  run's biggest losing tickers (see Factor 8).

### 7. Signal generation & strategy filtering

`KairosOrchestrator._apply_meta_filters` (entropy > 3.0, bimodality/kurtosis < -1.0),
the per-strategy `KurtosisFilterStrategy`/`LiquidityFilterStrategy` wrappers
(`kurtosis_max=10.0`, `min_volume_percentile=10.0`), the ~18 permanently-disabled
strategies plus per-profile disabled sets, and the `min_ev_pct=0.10%` gate all
determine which signals are even allowed to reach allocation. Given Factor 3 shows
realized costs land right at 0.15%, a signal only needs `ev_pct > 0.10%` to be
considered — a margin thinner than the realized cost itself.

**4.1 Statistic to optimize:** per-strategy/per-asset EV-net-of-realistic-cost
(not the theoretical 0.15% assumption but the *measured* per-asset-class cost from
Factor 3), and oracle Sharpe as the strategy-level upper bound already used by
`resolve_disabled_strategies`.

**4.2 Concrete changes:**
- Raise `min_ev_pct` above 0.10% (e.g., closer to or above the empirically-measured
  0.15% realized cost) so only signals that clear costs with real margin are taken —
  right now a signal can pass the gate with an EV edge *smaller* than what it will
  actually pay in costs.
- Re-run the oracle sweep behind `resolve_disabled_strategies` on recent data,
  specifically checking whether the top losing tickers from this run (`LDO-USD`,
  `AAVE-USD`, `BTC-USD`, `FIL-USD`, `ATOM-USD`, `XTZ-USD`, `AXS-USD`) should have more
  strategies disabled for them specifically, rather than relying on the current
  interval/asset-class-level defaults.

### 8. Underlying model/signal fitness

The base Kronos model's raw predictive quality per asset, and the gap between it and
the oracle (perfect-foresight) ceiling, is the fundamental limit on what any
downstream tuning (Factors 1-7) can achieve. `oracle_sharpe`/`oracle_win_rate` are
already computed and stored per strategy/asset in `viability_report` — they're the
right yardstick for where finetuning effort would pay off most.

**4.1 Statistic to optimize:** the oracle-vs-realized Sharpe gap, per asset, especially
for the tickers that actually dominate this run's losses.

**4.2 Concrete changes:**
- Prioritize finetuning specifically for `LDO-USD`, `AAVE-USD`, `FIL-USD`, `ATOM-USD`,
  `XTZ-USD`, `AXS-USD` (the top losing tickers here, none of which — except indirectly
  via `BTC-USD` — currently have an accepted finetuned model or are in
  `kairos_idle_finetune.py`'s default rotation), instead of continuing to only rotate
  through the hardcoded `BTC-USD/ETH-USD/SOL-USD` list.
- Feed `select_finetune_candidate`'s ranking (already in `kairos_pipeline.py`) with a
  wider `min_signals` sample if these tickers currently don't have enough oracle-viable
  strategies to be considered — otherwise they may simply never surface as finetune
  candidates even though they're the biggest realized drag.
