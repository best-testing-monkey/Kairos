# Kairos: Pluggable Prediction-Usage Modes (Architecture)

**Version:** 0.1
**Date:** 2026-09-19
**Target:** implementation
**Scope:** a build-time hook that controls what `AssetPrediction` a strategy receives, with a
registry of interchangeable modes. Zero changes to `generate_signal()`'s signature or to any of
the ~137 existing strategies. Companion to `DESIGN_DOC_ml_tpsl_optimization.md` (which makes
*stop/target determination* pluggable) — this document makes *prediction consumption* pluggable,
a different, orthogonal axis of the same problem: Kairos is about to have more than one way to
turn a model forecast into something a strategy acts on, in two independent places.

---

## 1. Motivation

Confirmed by reading current source (§2): today there is exactly one way a prediction becomes
usable by a strategy, and it isn't documented as a "mode" anywhere — it's just what the code
happens to do. Most strategies decide direction by comparing a single scalar
(`dist.stats["close"]["mean"]`) against `current_price`; a few gate that decision with a
technical indicator computed over real historical bars only. The predicted distribution never
touches the bar series indicators actually run over.

That was a reasonable default, but it's now one point in a design space, not the only option —
the same way "percentile of the distribution" was one point in the stop/target design space
before `DESIGN_DOC_ml_tpsl_optimization.md`. The immediate need: a second mode,
`distribution_as_bar` (own document, `DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md`),
where the forecast gets folded into the bar series itself so indicator-based strategies react to
it directly. This document is the generic hook that mode plugs into — written so a third mode
later doesn't need its own plumbing change.

## 2. Current state

| Fact | Where | Detail |
|---|---|---|
| Strategy call signature | `kairos_backtest.py:588-594` (base `Strategy.generate_signal`) | `(self, dist: KairosDistribution, current_price: float, history: pd.DataFrame, context: Dict, **kwargs)` — 4 args, one `AssetPrediction` unpacked into 3 of them. Matched by ~130 concrete strategies in `kairos_backtest.py`/`kairos_crypto.py`. |
| `AssetPrediction` | `kairos_meta.py:116-122` | Exactly 4 fields: `symbol`, `dist`, `current_price`, `history`. No `.samples`/`.distribution` alias. |
| Build site, model path | `kairos_strategies.py:582-586` (`predict_all_batch`) | `history=assets[symbol]` — raw input DataFrame, verbatim, up to `as_of`. No concat with the prediction. |
| Build site, oracle/naive path | `kairos_orchestrator.py:1028-1033` (`_make_realized_predictions`) | `history` is either untouched (oracle) or truncated by exactly one row (naive, `history.iloc[:-1]`, L1001-1002) — never extended. |
| Unpack + dispatch | `kairos_orchestrator.py:1118-1121, 1142` (`_run_day`) | `current_price = pred.current_price; dist = pred.dist; history = pred.history`, then `strat.generate_signal(dist, current_price, history, context)`. |
| How direction is actually decided today | `TrendFollowingStrategy` (`kairos_backtest.py:793-802`), `RSIFilterStrategy` (`:1161-1171`), `MACDFilterStrategy` (`:1213-1223`) | Direction = `sign(dist.stats["close"]["mean"] - current_price)`. Some strategies AND-gate this with an indicator computed on real `history` (RSI/MACD via pandas `.ewm()`/manual calc) — a confirmation check, never a merged bar series. |
| Closest existing precedent for "swap the forecast basis" | `_make_realized_predictions()` (`kairos_orchestrator.py:964-1034`) | Builds a substitute `.dist` via `KairosDistribution.from_bar()` from a chosen real bar (the true next bar for oracle, the withheld last bar for naive). Only ever *replaces* `.dist` or *truncates* `.history` — never appends a row to `.history`. |
| Mode-switching precedent | `OrchestratorConfig.no_prediction`/`.naive_baseline` (`kairos_orchestrator.py:352-412`), branch at `_run_day():1093-1097` | One `if/else` picks which function builds `Dict[symbol, AssetPrediction]`. Everything downstream — context enrichment, the strategy loop — consumes the result identically regardless of mode. Strategies are unaware which mode built it. **This is the template to copy.** |
| Documented design intent for "prediction as next bar" | searched `strategy/*.md`, `docs/`, docstrings | **None found.** `strategy/README.md:199` only describes `KairosDistribution` sample rows as "one Monte Carlo sample of the next bar" — descriptive, not a design statement about indicator consumption. The mental model of "strategies already use the prediction as the latest bar" does not match current code. |

## 3. Design

**New config field**, `OrchestratorConfig`:
```python
prediction_usage_mode: str = "last_real_bar"
```
`"last_real_bar"` is the identity/default — today's behavior, byte-for-byte.

**New registry**, alongside where strategies/filters are already registered (mirror
`kairos_strategies.py`'s existing strategy-registry pattern):
```python
PredictionUsageFn = Callable[[AssetPrediction], AssetPrediction]

PREDICTION_USAGE_MODES: Dict[str, PredictionUsageFn] = {
    "last_real_bar": lambda pred: pred,            # identity
    # "distribution_as_bar": <registered by its own module, see companion doc>
}
```

**Single insertion point**, `_run_day()` (`kairos_orchestrator.py`), immediately after
`multi_preds` is built (today's L1093-1097 branch) and before context enrichment / the strategy
loop:
```python
if self.config.no_prediction:
    multi_preds = self._make_realized_predictions(date, histories, naive=self.config.naive_baseline)
else:
    multi_preds = self.multi_predictor.predict_all(histories)

usage_fn = PREDICTION_USAGE_MODES[self.config.prediction_usage_mode]
multi_preds = {sym: usage_fn(pred) for sym, pred in multi_preds.items()}
```
This is one additive transform applied *after* whichever function built the base
`AssetPrediction` — so it composes with the oracle/naive axis automatically instead of needing
its own copy of that branch. A mode function receives a complete `AssetPrediction` and returns
one (new or mutated) with `.history`/`.current_price` adjusted as needed; `dist` is normally left
alone (the forecast itself doesn't change, only how much of it gets exposed to indicator logic).

**Why the hook lives here and not inside `Strategy.generate_signal`:** every strategy already
receives `dist`/`current_price`/`history` as plain, already-resolved values — there's no shared
"bracket calculator" to hook once (per `DESIGN_DOC_ml_tpsl_optimization.md`'s equivalent finding
for stop/target), but there *is* a single choke point where every strategy's inputs get built:
the `AssetPrediction` construction step. Hooking there means the ~137 strategies need zero
changes, exactly like the existing `no_prediction`/`naive_baseline` flags already achieve for the
oracle/naive axis.

**Two independent axes, now three total, all orthogonal:**

| Axis | Where it acts | Config |
|---|---|---|
| Evaluation basis (existing) | which function builds `.dist`/`.history` in the first place | `no_prediction`, `naive_baseline` |
| **Prediction usage (this doc)** | transforms the built `AssetPrediction` before strategies see it | `prediction_usage_mode` |
| TP/SL determination (companion doc) | wraps a strategy's *returned* `Signal`, overrides `.stop`/`.target` | wrapper strategy choice |

These can combine freely — e.g. live model + `distribution_as_bar` entry logic + ML-predicted
stop/target, or oracle-mode + `last_real_bar` (today's default) for a clean evaluation baseline.
No combination is disallowed by this design; some combinations (e.g. `distribution_as_bar` under
naive mode, where the "distribution" is itself a withheld-bar placeholder) are unusual but not
incoherent — flag as untested rather than forbidden.

## 4. Non-goals

- Does not implement `distribution_as_bar` itself — companion document.
- Does not change `generate_signal()`'s signature, or any strategy's internals.
- Does not change stop/target determination — separate axis, separate document.
- Does not retrofit the scalar `dist.mean`-vs-`current_price` direction check that most strategies
  use — those strategies keep behaving exactly as before under every mode unless the mode itself
  changes something they read (see companion doc §3 for which strategies are actually affected).

## 5. Risks

- **`_run_day()` is a hot, previously leak-and-crash-prone path** (see this repo's CLAUDE.md
  prewarm sagas). The insertion point must be a small additive dict-lookup + comprehension, not a
  restructuring of the function.
- **Default-mode regression is the load-bearing guarantee.** `prediction_usage_mode="last_real_bar"`
  must produce byte-identical `AssetPrediction`s (and therefore byte-identical signals) to current
  code — the identity function is trivial, but the *wiring* around it (config plumbing, the
  registry lookup) must not accidentally touch `dist`/`history`/`current_price` even under the
  default. Same discipline the 2026-08-29 exit-rule resweep used to confirm old vs. new rule
  parity.
- **New lookahead risk is the companion mode's concern, not this hook's** — but this hook is the
  place a future third mode could introduce one, since it's the one place with write access to
  `.history`/`.current_price` before strategies run. Any new mode added to the registry should
  carry its own no-lookahead unit test, given this project has hit this exact bug class twice
  (naive-baseline zero-drift trap; the oracle-decision peek bug, both documented in this repo's
  CLAUDE.md).

## 6. Testing plan

| Test | Setup | Pass criteria |
|---|---|---|
| Default-mode parity | Run `_run_day()` with `prediction_usage_mode="last_real_bar"` vs. current code (pre-change) on identical fixture data | Byte-identical `AssetPrediction`s and resulting `Signal`s |
| Registry dispatch | Register a dummy mode function, select it via config | `_run_day()` calls it, downstream strategies see its output |
| Composability with oracle/naive | Run `no_prediction=True, naive_baseline=True` together with each registered `prediction_usage_mode` | No exception; transform applies to the oracle/naive-built `AssetPrediction` exactly as it would to the model-built one |
| Unknown mode string | Config with a typo'd mode name | Raises clearly at startup (`KeyError`/explicit validation), not a silent no-op |

## 7. Implementation order

1. Add `prediction_usage_mode` field to `OrchestratorConfig`, default `"last_real_bar"`.
2. Add the `PREDICTION_USAGE_MODES` registry with just the identity entry; wire the insertion
   point in `_run_day()`.
3. Default-mode parity test (§6) — prove zero behavior change before anything else lands.
4. Hand off to the companion document to register `distribution_as_bar` as the second entry.
