# Kairos: Prediction-Usage Mode — `distribution_as_bar`

**Version:** 0.1
**Date:** 2026-09-19
**Target:** implementation
**Scope:** one new entry in the `PREDICTION_USAGE_MODES` registry defined in
`DESIGN_DOC_prediction_usage_modes_architecture.md`. Requires that document's hook to already
exist. No changes to strategies, no changes to `generate_signal()`.

---

## 1. Motivation

Today, indicator-based strategies (RSI/MACD/SMA-style — `RSIFilterStrategy`,
`MACDFilterStrategy`, others in `kairos_backtest.py`) compute their indicators over real
historical bars only, then AND-gate the result against a *separate* scalar check
(`dist.stats["close"]["mean"]` vs `current_price`). The forecast never reaches the indicator math
itself — an RSI reading never reflects what the model thinks happens next, only what already
happened.

`distribution_as_bar` folds the forecast into the bar series: build one synthetic OHLCV row from
the predicted distribution and append it to `history` as if it were the newest real bar. Every
indicator a strategy already computes over `history` then automatically incorporates the
forecast — no strategy code changes, per the companion architecture document's whole point.

## 2. Mechanism

**Inputs available**: `dist` (`KairosDistribution`, `.stats` has `open`/`high`/`low`/`close` keys
only — confirmed by reading `_compute_stats()`, `kairos_backtest.py:284-341`; `volume` is **not**
in `.stats` despite being a column in the raw samples, per the class docstring "open, high, low,
close, volume, amount" and `_compute_stats()`'s `for col in ["open","high","low","close"]` loop —
`dist.df["volume"]` exists raw and un-aggregated if needed), `current_price`, `history`.

**Synthetic bar construction**:
```python
def _build_synthetic_bar(pred: AssetPrediction) -> pd.Series:
    s = pred.dist.stats
    last_ts = pred.history.index[-1]
    next_ts = _advance_one_interval(last_ts, interval)   # reuse existing interval-stepping
                                                           # helper -- check kairos_signals.py /
                                                           # the multi-interval design doc before
                                                           # writing new date math
    return pd.Series({
        "open":  pred.current_price,             # continuity: picks up where real series left off
        "high":  s["high"]["pct_50"],             # median, not mean -- robust to the fat tail this
        "low":   s["low"]["pct_50"],               # repo already filters on via kurtosis_max
        "close": s["close"]["pct_50"],
        "volume": pred.history.iloc[-1]["volume"],  # carry-forward, see "Open question" below
    }, name=next_ts)
```

**Applying it** (the mode function registered in `PREDICTION_USAGE_MODES`):
```python
def distribution_as_bar(pred: AssetPrediction) -> AssetPrediction:
    bar = _build_synthetic_bar(pred)
    new_history = pd.concat([pred.history, bar.to_frame().T])
    return dataclasses.replace(pred, history=new_history)   # current_price and dist UNCHANGED
```

**`current_price` is deliberately left untouched.** You can only actually enter a trade at the
real current price — the synthetic bar exists so `history`-based indicators react to the
forecast, not so the entry price becomes a hypothetical future close. This is a design decision,
not an incidental detail: getting it backwards (setting `current_price` to the synthetic bar's
close) would mean sizing/entry math silently starts pricing off a number nobody can actually
trade at, a much bigger change than intended and one that would need to flow through
`allocation.py`'s entire Risk%/Reward%/Kelly chain (see `DESIGN_DOC_ml_tpsl_optimization.md` §2).

**Open question — volume.** `dist.stats` has no volume percentiles today (§2 above). Two options:
- **v1 (recommended): carry the last real bar's volume forward unchanged.** Simple, doesn't
  fabricate a volume-percentile pipeline that doesn't exist, and volume-dependent logic
  (`min_volume_percentile` filter, `LiquidityFilterStrategy`'s `scipy.stats.percentileofscore`
  call — see this repo's CLAUDE.md "scipy missing" gotcha) keeps working against a real number.
- **Future enhancement**: compute a percentile/mean directly from `dist.df["volume"]` (the raw,
  un-aggregated column exists) and add a `"volume"` entry to `_compute_stats()`'s per-column loop.
  Deferred — no current consumer needs a *predicted* volume percentile specifically, only a
  present value for indicators/filters that read `history["volume"]`.

## 3. Who this actually changes behavior for

Not universal. `TrendFollowingStrategy` (`kairos_backtest.py:793-802`) and the seven strategies
that alias it (per this repo's own "Eight strategy names are the same strategy" note) compute
direction purely from `dist.stats["close"]["mean"]` vs `current_price` — they never read
`history` for direction at all, so this mode has **zero effect** on them. It only changes
strategies that compute an indicator *over `history`* — `RSIFilterStrategy`, `MACDFilterStrategy`,
and any other strategy calling `.ewm()`/rolling stats/manual indicator math on the `history`
DataFrame. Worth stating plainly so nobody expects a blanket behavior shift from flipping the
mode — audit which strategies actually read `history` before estimating impact.

## 4. Lookahead-safety note

`dist` — regardless of whether it came from the live model, oracle peek, or naive withhold — is
already what today's code treats as "the forecast": strategies already read `dist.stats` directly
for direction. Appending its median as a `history` row re-presents the *same* information through
a different channel (indicator functions instead of a scalar comparison); it does not, by
construction, expose anything a strategy couldn't already see via `dist`. That said, this project
has hit exactly this bug shape twice (the naive-baseline zero-drift trap; the oracle-decision peek
bug — both in this repo's CLAUDE.md, 2026-08-28/2026-09-01) purely by getting the "what's visible
when" boundary subtly wrong, so this claim must ship with a unit test, not just this paragraph:
assert `_build_synthetic_bar`/`distribution_as_bar` reads nothing beyond the `AssetPrediction`
passed in (no reaching into `full_df`, no module-level state), and assert the resulting `history`
is otherwise identical to the input except for the one appended row.

## 5. Non-goals

- No volume-percentile modeling in v1 — carry-forward only (§2).
- No multi-bar-ahead synthetic history — exactly one synthetic bar, matching this project's
  existing `pred_len=1` assumption elsewhere (`kairos_predcache.make_key()`'s docstring notes the
  same constraint for a different cache).
- Does not change the default mode — `last_real_bar` stays default; this is opt-in via
  `prediction_usage_mode="distribution_as_bar"`.
- Does not touch which strategies exist or how they read `history`/`dist` — purely an input-shape
  change at the `AssetPrediction` level.

## 6. Testing plan

| Test | Setup | Pass criteria |
|---|---|---|
| Synthetic bar values | Fixture `dist.stats` with known percentiles | Returned bar's `open`/`high`/`low`/`close` match expected `current_price`/`pct_50` values exactly |
| `current_price` untouched | Any fixture | `distribution_as_bar(pred).current_price == pred.current_price` |
| History append-only | Any fixture | Returned `history` equals input `history` plus exactly one trailing row; no existing rows mutated |
| No-lookahead | Fixture where `dist` is built from a real future bar (oracle-style) | Function reads only `pred.dist`/`pred.history`/`pred.current_price` — no access to any other future data (assert via a mock/spy that fails on out-of-scope reads) |
| Indicator sensitivity | Feed `RSIFilterStrategy`/`MACDFilterStrategy` a history with and without the synthetic bar, same `dist` | RSI/MACD values differ, confirming the forecast now reaches indicator math |
| No effect on non-history strategies | `TrendFollowingStrategy` under both modes, same `dist`/`current_price` | Identical `Signal` output — confirms §3's scoping claim |

## 7. Implementation order

1. `_build_synthetic_bar()` + unit tests (§6, first 3 rows) — pure function, no wiring yet.
2. Register `distribution_as_bar` in `PREDICTION_USAGE_MODES` (from the architecture doc).
3. No-lookahead test (§6) — must pass before this is considered safe to enable anywhere.
4. Indicator-sensitivity + no-effect tests (§6) — confirm the scoped-impact claim in §3.
5. Offline comparison (reuse `kairos_signal_replay.py`-style tooling per
   `DESIGN_DOC_ml_tpsl_optimization.md`'s precedent) — do indicator-gated strategies actually
   perform differently, before considering this for live use.
