# Kairos Offline Signal Replay: Fast Selection & Allocation Iteration

**Version:** 1.2
**Date:** 2026-08-08
**Target:** Kimi Code implementation
**Scope:** Two new `pipeline_results.db` tables, a closure-stats computation pass, and an
offline allocation-replay CLI. No changes to signal generation, strategies, or the discovery
pipeline. No changes to `kairos_papertrade.py`'s live execution path. **Unleveraged only** —
see the scoping note below and §4.
**Interval-agnostic:** nothing in this design assumes daily bars. A "day" below is shorthand
for "one signal-generation timestamp at whatever interval is in play" — 1h, 4h, 1d, or a mix,
per §3.2's per-signal interval ladder. Where the text below still says "day"/"daily" read it as
"replay step," not as a hardcoded calendar-day assumption.

### Phase scope: unleveraged only

This phase is **cash/spot only**. `max_leverage` is fixed at `1.0` throughout — no margin
locking, no `admission_check` gating, no liquidation simulation, no CFD margin-class handling.
The whole margin/leverage/MTM epic from earlier this session (`docs/tickets/DESIGN_DOC_mtm_margin_leverage.md`)
is a live-execution-path concern; this document's replay loop does not touch it. A future phase
could extend §3.4's replay loop to simulate leverage once the unleveraged case is validated and
trusted, but implementing that is explicitly **not** this document's job — see §4.

---

## 1. Motivation

The discovery pipeline (`kairos_pipeline.py`) already proves, independently:

- a strategy is viable under perfect (oracle) predictions,
- a strategy is viable in combination with the base model's predictions,
- a strategy is viable in combination with a finetuned model's predictions.

(`viability_report`'s `oracle_*`/`base_*` columns — see `strategy/kairos_pipeline.py`, `docs/tickets`
of the earlier discovery-pipeline epics.)

What is *not* proven, and what the two live 6-month benchmark runs on 2026-08-07 showed is
actively losing money on the honest (MTM) curve even after fixing a real execution-layer bug
(`docs/tickets/RESEARCH-01-sl-close-reason-positive-pnl.md`): **which signals, at any given
replay step, should actually be traded, and how much capital each should get.** That is
entirely `strategy/allocation.py`'s job (`fetch_signals`/`allocate`, the `--signal-selection`
DSL, Kelly sizing, `max_pos_pct`/`max_cluster_pct`/`gross_cap_pct`). Leverage/margin fields on
`AllocationConfig` are out of scope for this phase — see the scoping note above and §4.

Today, testing a new selection rule or allocation formula means running
`kairos_papertrade.py` end-to-end — real model inference, real price fetches,
`phantom_ledger` fills — for hours, per idea. That is not an iteration loop; it is a queue.

This document specifies an **offline replay path**: precompute, once, what each individual
signal *would have done* if traded in isolation (its own realized outcome — no portfolio
interaction), then let a separate, fast, pure-Python replay loop try arbitrary selection/
allocation rules against those precomputed outcomes in seconds, not hours.

### The one thing this document exists to prevent

Tonight's root cause (`RESEARCH-01`) was two independent local data sources silently
diverging — a signal cached against a stale price, executed later against a fresher one — and
it fabricated an entire fake +26% return undetected until a live run was actually driven and
inspected. A fast offline replay tool is a *second, structurally different* place the same
failure shape can recur: if the closure-stats computation and the live execution path use
different price data, different cost assumptions, or different fill logic, this tool will
happily produce a confident, wrong, fast answer instead of a confident, wrong, slow one. Every
design decision below is made with that risk explicitly in view, not as an afterthought.

---

## 2. Current state (what this builds on)

| Existing piece | Location | Relevance |
|---|---|---|
| Per-strategy signal cache | `kairos_signals.py`'s `signals_cache` table (`pipeline_results.db`) | Already stores `stats_json`/`advice_json` — the raw per-`as_of`-timestamp, per-group signal rows, at whatever `interval` that run used (already interval-parameterized today, not daily-only) — keyed by `(strategy_name, assets, interval, as_of_date, lookback, pred_samples, min_ev_pct, model_path, checkpoint_fingerprint)`. This is the *input* signal source; see §4.1 for why it isn't reused as-is for the new tables. |
| Fast, predictor-free exit/PnL primitives | `strategy/kairos_backtest.py`'s `BacktestEngine._check_exit`/`_calculate_pnl` (private methods; `Trade` is the dataclass shape closure rows are modeled on, but is NOT itself produced by this reuse path — see §3.3) | `_check_exit(position, bar)`/`_calculate_pnl(position, exit_price)` resolve a GIVEN position (entry/stop/target/direction already decided) against one bar at a time — no `phantom`, no network beyond the price fetch itself, no GPU, and critically **no `KairosPredictor`/`DecisionTreeRouter`** (unlike `BacktestEngine.run()`, which needs both to generate its own signals — confirmed by reading current source, do not assume `.run()` itself is reusable here). Cost as a flat `fee_pct`/`slippage_pct` (default `0.0005`), **not** `phantom`'s per-broker-profile model (commission floor, dynamic spread, fx conversion — see `phantom/profiles/ibkr.json`, exercised in `tests/unit/test_kairos_papertrade_leverage_regression.py`'s hand-derivations this session). This is what §3.3 reuses for closure computation — see the explicit cost-model caveat there. |
| Live, ground-truth execution | `strategy/kairos_papertrade.py`'s day loop (already `--interval`-parameterized, not daily-only) | The real thing: `phantom_ledger` fills/SL-TP, `_IntradayFallbackProvider` price fetches, the full margin/MTM epic from this session, and (as of tonight) the stale-bracket guard from `RESEARCH-01`. Slow (hours), authoritative. Nothing here changes it. |
| Selection & allocation | `strategy/allocation.py`'s `fetch_signals`/`allocate`, `AllocationConfig`, `--signal-selection` DSL (`strategy/signal_selection.py`) | The actual thing under test. Reused UNCHANGED by the offline replay loop (§4.4) — the whole point is to try different `AllocationConfig`/selection-rule values against the same precomputed outcomes, not to reimplement allocation logic a second time. |

---

## 3. Design

### 3.1 New tables (`pipeline_results.db`)

Two tables, mirroring `signals_cache`'s own cache-key/fingerprint discipline (per
`kairos_signals.py`'s `_signals_cache_key()` — see CLAUDE.md's "Per-strategy signals cache"
section) so re-running the precompute pass over an already-covered window is a cheap no-op,
not a recompute.

```sql
-- One row per INDIVIDUAL signal (unpacked from signals_cache's per-group JSON blobs),
-- not per (group, date) batch. This is the normalization signals_cache doesn't do.
CREATE TABLE IF NOT EXISTS papertrade_signals (
    signal_id       TEXT PRIMARY KEY,   -- deterministic hash of the fields below
    strategy_name   TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    direction       TEXT NOT NULL,      -- "long" | "short"
    interval        TEXT NOT NULL,      -- the interval THIS SIGNAL was generated at (1h/4h/1d/...)
    as_of           TEXT NOT NULL,      -- signal-generation timestamp, at the granularity of `interval` above
    entry           REAL NOT NULL,
    stop            REAL NOT NULL,
    target          REAL NOT NULL,
    expected_value  REAL,
    base_win_rate   REAL,
    n               INTEGER,            -- base_signals/oracle_signals fallback, per fetch_signals()
    model_label     TEXT NOT NULL,      -- "base" | finetuned group label
    checkpoint_fingerprint TEXT NOT NULL DEFAULT '',
    source_cache_key TEXT,              -- signals_cache.cache_key this was unpacked from, if any
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papertrade_signals_as_of ON papertrade_signals(as_of);

-- One row per papertrade_signals row: the signal's OWN, ISOLATED outcome -- what happens
-- if this single signal is traded alone, ignoring every other concurrent signal, cap, or
-- capital constraint. Portfolio-level effects are the replay loop's job (§3.4), not this
-- table's.
CREATE TABLE IF NOT EXISTS papertrade_signals_closure (
    signal_id           TEXT PRIMARY KEY REFERENCES papertrade_signals(signal_id),
    resolved            INTEGER NOT NULL,   -- 0 if disqualified (see §3.2); every other
                                             -- column is NULL when resolved=0
    interval_used       TEXT,               -- the interval that actually succeeded (§3.2)
    pct_profit          REAL,               -- Trade.pnl_pct from BacktestEngine
    max_drawdown_pct     REAL,              -- worst adverse excursion DURING the trade's own
                                             -- life (MAE-style), direction-aware -- NOT a
                                             -- portfolio drawdown, see §3.3
    trigger_datetime     TEXT,               -- Trade.entry_date
    exit_datetime         TEXT,               -- Trade.exit_date
    exit_reason           TEXT,               -- Trade.exit_reason ("tp"/"sl"/etc.)
    computed_at           TEXT NOT NULL,
    engine_version        TEXT NOT NULL       -- bump when BacktestEngine's cost/fill logic
                                             -- changes, to force recompute -- see §3.2's
                                             -- reuse of signals_cache's invalidation pattern
);
```

`papertrade_signals` is deliberately a cache in front of `signals_cache`, not a replacement for
it: if a `(strategy, ticker, interval, as_of, model_label, checkpoint_fingerprint)` row already
exists, the precompute pass skips re-unpacking it from `signals_cache`. `papertrade_signals_closure`
is a cache in front of the `BacktestEngine` call, keyed 1:1 on `signal_id` plus `engine_version`
(mirroring `checkpoint_fingerprint`'s role in `signals_cache`/`kairos_predcache`: a fixed key
component that busts the cache when the thing that would change the *meaning* of a cached value
changes, even though it's not part of what identifies the signal itself).

### 3.2 Interval selection & disqualification

Per the confirmed decision: **smallest available interval, chosen per (ticker, signal) —
not globally.** Different tickers can have different data availability at any given interval
(exactly the kind of per-ticker gap `RESEARCH-01`'s `no_data_tickers` mechanism already tracks
for daily bars).

```python
def resolve_closure(signal: PapertradeSignal, interval_ladder: list[str]) -> ClosureResult:
    """Try intervals smallest-first. First one with sufficient price data for this
    signal's ticker over its resolution window wins. If none succeed, the signal is
    DISQUALIFIED -- no closure row is written with resolved=1, no stats are estimated
    or interpolated. A disqualified signal is invisible to the replay loop (§3.4), not
    included with a null/zero outcome -- silently treating "no data" as "no profit" would
    bias every allocation experiment run against this table, the same class of silent
    corruption RESEARCH-01 just spent a session finding.
    """
```

This is a hard requirement, not an optimization: no interpolation, no fallback to a coarser
interval's OHLC pretending to be finer-grained, no "assume flat" placeholder. Disqualified
means absent.

### 3.3 Closure computation (reuses `BacktestEngine`'s exit/PnL primitives, not `phantom`)

Per the confirmed decision: reuse `strategy/kairos_backtest.py`'s execution/cost logic, not
`kairos_papertrade.py`'s `phantom_ledger` path. This is the fast/cheap choice, and it is a
**deliberate, explicit divergence from the live execution path's cost model** — but it is
**not** a call to `BacktestEngine.run(df, router, ...)` end-to-end.

**Why not `.run()` — precisely, not just "it needs a predictor":** `run()`'s job is to
*generate* a signal from raw price history via `dist = self.predictor.predict(history)` then
`router.route(dist, ...)` (`strategy/kairos_backtest.py` ~line 1930-1940) — deciding whether to
enter at all, and at what stop/target. Closure computation doesn't need that decision made: the
signal (`entry`/`stop`/`target`/`direction`) is already sitting in `papertrade_signals`,
unpacked from a `signals_cache` row `kairos_signals.py` computed once, historically. The job
here is strictly "given this already-decided signal, what happened next" — a pure price-path
question with no prediction step in it at all. Note this is true independent of whether
predictions are cheap or expensive to obtain: `kairos_strategies.py`'s real production
`predict_kairos_cloud` (not the docstring/`__main__` stub inside `kairos_backtest.py` itself —
confirmed by reading current source, don't confuse the two) does check a cache before hitting
the GPU, but it's a small, per-process, in-memory `_prediction_cache` (capped at 5000 entries,
wiped every fresh process start) — a *different*, narrower cache than the persistent,
disk-backed `kairos_predcache` that `kairos_papertrade.py`'s prewarming actually populates under
`data/predcache/`. Even wired to the persistent cache, `.run()` would still be the wrong tool
here, because we'd be asking it to re-derive a decision we already have the answer to, not
because a prediction call would be slow.

What's actually reusable, and genuinely prediction-free, is `BacktestEngine`'s pair of private
exit/PnL helpers:

- `_check_exit(position: dict, bar: pd.Series) -> tuple[float | None, str | None]` — given a
  position dict (`direction`, `stop`, `target`, `entry`, ...) and ONE bar's OHLC, returns
  `(exit_price, exit_reason)` if that bar resolves the trade (open-gap-through checked first,
  then intrabar stop/target touch, else `None, None` to keep walking). Pure, no predictor.
- `_calculate_pnl(position: dict, exit_price: float) -> float` — gross P&L minus a flat
  `fee_pct` charge, again no predictor.

Both are called from a `BacktestEngine` instance, but neither reads `self.predictor` — a
`BacktestEngine(predictor=None, fee_pct=..., slippage_pct=...)` instance can call them safely
as long as `.run()`/`.run_strategy_comparison()` are never invoked on it. Closure computation
therefore constructs one such instance and, given a signal's `entry`/`stop`/`target`/`direction`
plus a bar-by-bar price DataFrame from `entry_datetime` onward, walks bars calling
`_check_exit` on each until it resolves (or the available data runs out — see §3.2's
disqualification rule for that case), then `_calculate_pnl` for the final `pct_profit`.

These are underscore-prefixed (private) methods on `BacktestEngine`, not a published API.
Reaching into them anyway, with this documented justification, matches this codebase's existing
precedent for the same tradeoff — `kairos_papertrade.py` already reaches into `phantom`'s
private `_conn` when the public API doesn't cover an exact need (see
`remove_all_open_positions`'s docstring from this session's MTM epic). If a future refactor of
`kairos_backtest.py` renames or changes the behavior of `_check_exit`/`_calculate_pnl`, this is
the coupling that breaks — bump `engine_version` (§3.1) and re-verify against current source
when that happens, don't assume this document stays accurate forever.

Cost-model divergence from live execution, regardless of mechanism:

- This path costs a trade at a flat `fee_pct` + `slippage_pct` of notional (`BacktestEngine`'s
  own convention).
- `kairos_papertrade.py` (via `phantom`) costs a trade at a per-instrument-class model:
  per-share commission with a floor, a spread model, and (per `RESEARCH-01`'s investigation
  this session) an entry-only fx conversion charge for non-USD-base accounts.

These will not agree to the cent, and are not meant to. The offline tool answers "does this
selection/allocation rule look directionally better than that one," not "what will the live
P&L be." **Any rule that looks promising here must still be validated with a real
`kairos_papertrade.py` run before being trusted** — this is a non-goal boundary (§4), not an
implementation detail to fix later, because closing that gap means reimplementing phantom's
cost engine a second time, which is exactly the kind of duplicated-source-of-truth problem
`RESEARCH-01` traces back to.

`max_drawdown_pct` (the per-signal, isolated max adverse excursion) is not something
`_check_exit`/`_calculate_pnl` compute — it requires walking the SAME bars the exit-resolution
loop above already walks and additionally tracking the worst mark-to-entry excursion at each
bar, direction-aware (same formula shape as `kairos_mtm.unrealized_pnl`, just evaluated
bar-by-bar and taking the worst value instead of only the final one). This is new code inside
the closure-computation pass, most naturally computed in the SAME bar-walking loop that calls
`_check_exit` each iteration (one loop, two things tracked), not a separate pass over the data.
Verify current source before implementing — this document describes source read on 2026-08-08,
and a Kimi implementer should confirm against the actual installed `kairos_backtest.py`, not
trust this document as permanently accurate.

### 3.4 Offline allocation replay loop

The actual point of the exercise. Given `papertrade_signals`/`papertrade_signals_closure`
populated over a window, replay a **candidate `AllocationConfig`/selection rule**, stepping
through **whichever distinct `as_of` timestamps actually have data**, not a fixed calendar-day
grid:

```python
def replay(start_ts, end_ts, interval: str, alloc_config: AllocationConfig,
           selection_rule: SignalSelectionRule | None, starting_capital: float) -> ReplayResult:
    """Pure-Python, no phantom, no GPU, no network. `interval` selects which
    papertrade_signals rows are in play (e.g. "1h" or "1d") -- a single replay run
    is scoped to one signal-generation interval, so results at different cadences
    aren't silently blended; running the same window at a different --interval is
    a separate invocation, not a flag combination within one.

    replay_steps = sorted(DISTINCT as_of FROM papertrade_signals
                           WHERE interval = :interval AND as_of BETWEEN :start_ts AND :end_ts)
    -- NOT range(start_ts, end_ts, timedelta(days=1)) -- the step grid is whatever
    -- timestamps this interval's signals actually landed on, which is 1-day-spaced
    -- for interval="1d" and NOT for interval="1h"/"4h". Deriving the grid from the
    -- data, rather than assuming a fixed cadence, is what makes this interval-agnostic.

    For each ts in replay_steps:
      1. Load every papertrade_signals row with as_of == ts AND a resolved closure row.
      2. Run them through allocation.fetch_signals()/allocate() UNCHANGED -- same code
         a live run uses -- to get this step's selected, sized candidates.
      3. Apply each selected candidate's PRECOMPUTED closure outcome (pct_profit,
         max_drawdown_pct, exit_datetime) to a running capital ledger, sized per
         allocate()'s output, using SIMPLE spot/full-notional cash bookkeeping only
         (mirrors kairos_papertrade.py's `--max-leverage 1.0` legacy path -- no margin
         lock, no admission_check, no liquidation_check; see the phase-scope note in §1).
      4. Track running equity/drawdown across the whole window.
    Returns a metrics dict shaped like a subset of compute_final_metrics()'s output
    (total_profit_eur, pct_profit, num_trades, pct_max_drawdown at minimum) so a
    replay result and a live papertrade result can be compared on the same axes.
    """
```

This loop is where portfolio-level effects (concurrent position caps, `gross_cap_pct`, capital
actually available at a given replay step) live — `papertrade_signals_closure` deliberately has
none of that, by design (§3.1), so this loop is the only place it's simulated, using the exact
same `allocate()` call a live run makes.

**Known simplification, flag it rather than hide it:** this replay loop sizes and "opens"
positions based on `allocate()`'s output at replay step N, then resolves them using the
closure's OWN isolated `exit_datetime` — it does not model a position closing early because a
*different*, correctly-sized-at-the-time reallocation event happened at an intermediate step.
For most strategies with independent per-position sizing this is a reasonable approximation;
for strategies that behave very differently under concurrent-position pressure, it will not be.
State this limitation in the tool's own `--help`/README, not just here.

### 3.5 CLI

New script or `kairos_pipeline.py --stage` addition (implementer's call — check which fits this
repo's existing CLI conventions better before choosing):

```bash
uv run ./strategy/kairos_signal_replay.py --precompute --months-back 6 --interval-ladder 1h,4h,1d
uv run ./strategy/kairos_signal_replay.py --replay --interval 1d --start 2026-02-06 --end 2026-08-07 \
    --signal-selection "'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %' DESC, TOP 3" \
    --capital 200 --max-pos-pct 15
```

`--precompute` populates the two new tables (idempotent, cache-aware per §3.1/§3.2) across
every interval in `--interval-ladder` — that flag is the per-signal fallback ladder from §3.2,
not a choice of one interval. `--replay` runs §3.4 against already-precomputed data for the
single `--interval` given and prints/writes a metrics summary — this is the fast loop, seconds
not hours, meant to be run many times with different `--signal-selection`/`AllocationConfig`
flag combinations. Comparing cadences (e.g. "does 1h beat 1d for this rule") means running
`--replay` once per `--interval` and comparing the resulting metrics — deliberately two
invocations, not one flag that blends both (per §3.4's scoping note).

---

## 4. Non-goals

- **No leverage, margin, or CFD simulation in this phase.** `max_leverage` is fixed at `1.0`
  everywhere in this document's scope. No `admission_check` gating, no margin-utilization caps,
  no liquidation modeling, no CFD margin-class handling — the entire margin/leverage/MTM epic
  (`docs/tickets/DESIGN_DOC_mtm_margin_leverage.md`) is out of scope here. If a leveraged
  offline-replay phase is wanted later, it is a separate design document, written after the
  unleveraged case is built, validated, and trusted — not a flag added to this one.
- **Not a replacement for live `kairos_papertrade.py` validation.** Per §3.3, the cost model
  and per-position portfolio interaction are both simplified. A selection/allocation rule that
  wins here is a *candidate*, confirmed only by an actual live run.
- **No interpolation or estimation for missing/disqualified signals** (§3.2) — silently biases
  every downstream experiment.
- **No changes to `kairos_papertrade.py`, signal generation, or the discovery pipeline.**
- **No new model inference** — closure computation consumes already-generated signal rows
  (`entry`/`stop`/`target`) and fetches price history for outcome resolution; it does not
  re-run Kronos.

---

## 5. Testing plan

| Test | Setup | Pass criteria |
|---|---|---|
| Interval ladder falls back correctly | Ticker with data at `1d` but not `1h` | `interval_used == "1d"`, not silently wrong |
| Full ladder exhausted | Ticker with no data at any configured interval | `resolved=0`, no row written with fabricated stats |
| Closure math matches hand-derivation | Synthetic price series, known entry/stop/target | `pct_profit`/`max_drawdown_pct`/`exit_reason` match a hand-computed value (same discipline as this session's `TestCorrectedCashFillCloseDelta`/`test_leverage_off_matches_pinned_baseline`) |
| Cache reuse | Run `--precompute` twice over the same window | Second run touches zero new rows, confirmed via row-count/`created_at` check |
| `engine_version` bump forces recompute | Bump the constant, rerun `--precompute` | Existing closure rows get recomputed, not silently left stale |
| Replay loop matches `allocate()` semantics | Feed a single replay step's worth of signals (any interval — test with both `1h` and `1d` fixtures) with a known `--signal-selection` rule | Same candidates selected/sized as calling `allocate()` directly with the same inputs |
| Replay step grid is data-driven, not calendar-day | `papertrade_signals` fixture at `interval="1h"` with non-daily-spaced `as_of` timestamps | Replay visits exactly those timestamps, not a synthesized daily grid |
| Replay vs. live sanity check (integration, best-effort) | Same window, same rule, one replay run + one real (possibly-mocked) papertrade run | Directionally similar sign/ranking, NOT expected to match in magnitude — document the expected gap, don't assert tight equality (would be a false invariant given §3.3's cost-model divergence) |

---

## 6. Implementation order

1. `papertrade_signals`/`papertrade_signals_closure` schema + `--precompute`'s unpack-from-`signals_cache` pass (no closure computation yet — just get signals normalized and cached).
2. Interval-ladder resolution + disqualification logic (§3.2), still no closure computation — prove the "which interval, which tickers get skipped" logic in isolation first.
3. Closure computation via `BacktestEngine` (§3.3), including the new max-drawdown-per-trade walk.
4. Replay loop (§3.4) reusing `allocate()` unchanged.
5. CLI (§3.5) + README/`--help` documentation of the non-goals in §4 (this is not optional polish — the cost-model gap needs to be visible to whoever runs this tool, not just in this document).
