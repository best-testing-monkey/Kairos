# Kairos: ML-Conditioned Stop/Target Placement

**Version:** 0.1
**Date:** 2026-09-19
**Target:** implementation, phased (offline validation first, live wiring is a later phase)
**Scope:** a new wrapper strategy that learns stop/target *distance* from historical outcomes,
plus the offline label-generation and validation tooling it needs. No changes to the ~137
existing `generate_signal()` implementations, no live `kairos_papertrade.py` wiring in phase 1.

---

## 1. Motivation

Confirmed by reading current source (see §2): almost every Kairos strategy sets its stop/target
by taking a fixed percentile of the model's 100 sampled future paths — typically the 15th/85th
percentile of predicted close, unconditionally. This number was never fit to historical outcomes;
it's a static default duplicated across dozens of `__init__` signatures. Because `allocation.py`
turns `Risk %`/`Reward %` (derived straight from stop/target) into the payoff ratio `b`, Kelly
sizing, and the ranking `Score`, an unoptimized bracket doesn't just cost a worse exit — it
corrupts which signals get selected and how big they're sized.

Goal: a cheap (no GPU, no deep learning) model that predicts a *better* stop/target distance,
conditioned on the situation, learned from Kairos's own historical resolved signals.

## 2. Current state

| Piece | Location | Relevance |
|---|---|---|
| `Signal` dataclass | `kairos_backtest.py:192-202` | `entry`/`stop`/`target` are absolute prices; `metadata: dict` is free-form, unread by sizing math — the provenance field a new wrapper should use. |
| `KairosDistribution` | `kairos_backtest.py:273-344` | Builds `pct_5..pct_95` percentiles from the model's 100 sampled paths. The dominant existing pattern (`stop = pct_15`, `target = pct_85`, e.g. `kairos_sentiment.py:210-311`) reads straight from this. |
| Other bracket approaches in use | fixed-% (`kairos_crypto.py:57-58`), ATR-multiple (`kairos_backtest.py:1037-1061`, defaults at `:1033`), structural/support (`kairos_backtest.py:1725-1760`), path-shape (`kairos_path.py:435-485`), horizon-hybrid (`kairos_horizon.py:309-345`), spread-relative (`kairos_forex.py:358-359`) | ~10 distinct approaches exist; none are learned. Confirms this is a systemic gap, not one strategy's bug. |
| **Existing wrap-and-override pattern** | `ATRBracketStrategy` (`kairos_volatility.py:170-245`), `VolumeProfileLevelsStrategy` (`kairos_execution.py:1273-1320`), `MetaLabelStrategy` (`kairos_ml.py:183-234`) | Call a base strategy, override `.stop`/`.target` on the returned `Signal`, record original values in `.metadata`. **This is the integration point** — a new ML bracket-picker wraps any base strategy the same way, instead of touching 137 files. |
| Downstream sizing | `allocation.py:128-187` | `risk_pct`/`reward_pct` (:147-148) → `b` (:151-156) → Kelly (:171-172) → `Score` (:174, the ranking key). Confirms the blast radius of a bad bracket. |
| Offline, GPU-free backtest resolution | `kairos_signal_replay.py`, reusing `BacktestEngine._check_exit`/`_calculate_pnl` | Already walks bars forward from a signal's `as_of` to resolve stop/target hits without touching the GPU or `phantom`. This is exactly the mechanism needed to compute hindsight labels (§4) and to validate a new model (§6) — reuse it, don't rebuild it. |
| Labeled historical outcomes | `oracle_results`/`signals_cache`/`strategy_class_stats` in `pipeline_results.db` | Per-signal resolved outcomes already exist via the 2026-08-29 walk-forward `_resolve_exit` fix. Per-class granularity exists too (`strategy_class_stats`), relevant to §7's data-imbalance risk. |

## 3. Prior art scan (external research, see agent report this session)

**Classical (non-ML), all well-established**: ATR-multiple stops, Chandelier Exit (trailing,
ratchets only), Donchian/Keltner channels, fixed R-multiples, support/resistance-based stops,
generic trailing stops, time-based (vertical-barrier) exits. All are variants of "distance
scaled by volatility, or by chart structure."

**ML-adjacent approaches, checked against literature:**
- **Triple-barrier method / meta-labeling** (López de Prado, *AFML*) — the barrier widths are
  **not learned end-to-end**. AFML sets them as `entry × (1 ± τ·σ)`, σ a rolling volatility
  estimate, τ chosen by a small **grid search validated with purged/embargoed CV** — and AFML
  explicitly warns that directly optimizing τ against backtest performance overfits. Meta-labeling
  itself is a secondary classifier that gates *whether to trade*, not where to place the barrier.
- **Stop-loss-aware labeling** (Hwang et al., *Finance Research Letters* 2023) — adjusts training
  labels for stop-loss interaction; the closest peer-reviewed paper to this problem, but still not
  barrier-width learning.
- **RL for dynamic exits** — active but immature; documented failure mode is regime instability
  (agents trained in calm windows learn stops too tight, high-vol windows too wide). Explicitly
  ruled out here anyway — not cheap, not stable.
- **Quantile regression / GBM (LightGBM `objective='quantile'`, XGBoost `reg:quantileerror`)** —
  the mechanism is real and well-supported for general price quantile forecasting, but **no
  rigorous published work applying it specifically to stop/target or barrier-width prediction was
  found**. This is open territory, not an established practice — closest in spirit to what Kairos
  already does manually with its own percentile mechanism (§2), but nobody's shown it beats the
  volatility-scaled baseline in print.
- **Bayesian optimization / genetic algorithms** — well-established for tuning a handful of
  *global static* SL/TP parameters against a backtest objective, but carries the same overfitting
  risk AFML warns about, amplified by a stronger search.
- **HMM regime-switching** — a different static rule per detected regime; a discretized form of
  volatility scaling, not a continuously-learned function.

**Verdict: fragmented, no established ML best-practice for this exact problem.** The one thing
sources agree on is that stop/target distance should be **conditioned on volatility/regime**, and
that **naively optimizing bracket width against backtest performance overfits** — both directly
shape §5 below.

**Recurring factors across sources** (used regardless of which method wins — this is the fallback
feature list): realized volatility/ATR, holding horizon, trend/momentum strength, volatility
regime, distance to support/resistance, liquidity/spread, asset class, **model prediction
confidence / distributional spread** (Kairos already computes this via `KairosDistribution`),
return skew/kurtosis (Kairos already filters on kurtosis), portfolio correlation, entry's position
within recent range.

## 4. Design

**Model — corrected during breakdown (2026-09-19), read before implementing.** The original draft
of this section specified LightGBM. Checked against actual source during ticket breakdown:
`pyproject.toml` has no `lightgbm`/`xgboost`/`scikit-learn` dependency anywhere, and
`kairos_ml.py`'s existing `GBMDirectionStrategy` already solves "cheap gradient-boosted trees
without a new ML dependency" with a **hand-rolled, dependency-free binary GBM**,
`GradientBoostedStumps` (`kairos_ml.py:373-430` — depth-2 trees, logloss/sigmoid objective,
numpy only, `fit(X, y)` / `predict_proba(X)`). Introducing LightGBM would duplicate a solved
problem and add a dependency this codebase has visibly avoided. **Reuse it instead of adding a
library**, reframed as a classification problem instead of quantile regression:

- Keep a small candidate grid of `(stop_pct, target_pct)` pairs (e.g. 3×3).
- Train **one `GradientBoostedStumps` instance per candidate pair**, predicting
  `P(target hit before stop | features)` — binary label, exactly the shape
  `GradientBoostedStumps` already supports unchanged.
- At inference: evaluate every trained candidate's `predict_proba`, compute
  `EV = p_win * reward_pct - (1 - p_win) * risk_pct` per candidate, pick the argmax.

This is not a downgrade of the idea — it's the same "predict distance conditioned on features"
goal, reached with zero new dependencies and code this repo already trusts. It also sidesteps
needing a single hindsight-best-width regression target (harder to get right) in favor of N
simpler, independently-checkable binary labels (§4.2, revised below). Predicts a **stop/target
distance** (`stop_pct`/`target_pct`, matching the existing `pct_X` convention strategies already
use), not an absolute price — still a drop-in replacement for the static
`stop_pct=15.0`/`target_pct=85.0` pattern, still avoids the AFML overfitting trap of hard-coding
one global optimum, since the argmax is per-signal and conditioned on that signal's own features.

**Features** (from §3's recurring-factor list, all either already computed in Kairos or cheap to
add):
- `KairosDistribution` stats already in hand: `entropy()`, `kurt`, std/skew per `dist.stats`
- ATR (already computed by `ATRBracketStrategy`/`kairos_horizon.py`, reusable)
- Recent realized volatility (rolling stdev of returns)
- Holding horizon / interval (`1h`/`4h`/`1d`)
- Asset class (`kairos_strategies.asset_class_for()` — reuse, don't reinvent per the "Four
  classifiers" caution in this repo's CLAUDE.md)
- Trend/momentum strength (already available context in most strategies)
- Distance to recent support/resistance (already computed by `SupportConfluenceStrategy`)

**Label (§4.2 detail, revised)**: for each historically resolved signal and each candidate
`(stop_pct, target_pct)` pair in the grid, walk that signal's actual subsequent price path via
`BacktestEngine._check_exit`/`_calculate_pnl` — reusing the exact mechanism
`kairos_signal_replay.py` already uses — and label that (signal, candidate) row `1` if target
would have been hit before stop, else `0` (unresolved candidates excluded, same discipline as the
2026-08-29 exit-rule fix documented in this repo's CLAUDE.md). This is a **hindsight label** built
from the future price path, which is correct and standard for supervised learning of outcomes —
but see §7's lookahead-leakage caution: the *features* fed to the model at train and inference
time must only use information available at signal time, never the resolved outcome or later
bars. This project has hit exactly this class of bug twice before (the naive-baseline zero-drift
trap, and the oracle-decision peek bug, both in CLAUDE.md's dated sections) — treat it as a known
project risk, not a hypothetical.

**Validation**: purged/embargoed cross-validation (AFML's explicit recommendation for this exact
parameter class), not a random split — adjacent-in-time signals share volatility-regime
information and will leak across a naive split.

**Integration**: a new wrapper strategy, same shape as `ATRBracketStrategy`/`MetaLabelStrategy` —
wraps any base strategy, evaluates every trained per-candidate `GradientBoostedStumps` classifier
for `(stop_pct, target_pct)`, picks the EV-argmax candidate, overrides `.stop`/`.target` on the
returned `Signal`, records the base strategy's original bracket plus the model's
inputs/output/chosen candidate in `.metadata` for auditability.

## 5. Non-goals (phase 1)

- No deep learning, no RL — ruled out by the "cheap" requirement and by §3's finding that RL exits
  are immature/unstable.
- No live `kairos_papertrade.py` wiring — validate offline first (§6), same phasing discipline
  `DESIGN_DOC_offline_signal_replay.md` used for allocation-rule changes.
- No rewrite of the ~137 existing strategies' internal bracket logic — this is an optional wrapper,
  applied selectively, not a replacement of the percentile mechanism everywhere.
- No cross-asset/portfolio-level bracket optimization — per-signal only, matching
  `kairos_signal_replay.py`'s existing per-signal-isolated scope.
- No global-constant retuning via backtest grid search — that's the exact overfitting pattern AFML
  warns against; if it's tempting as a quick win, it's the wrong quick win.

## 6. Testing / validation plan

| Test | Setup | Pass criteria |
|---|---|---|
| No feature leakage | Assert every feature column is computable from data available strictly before the signal's `as_of` timestamp | Static/code-review check + a unit test asserting feature-extraction never touches post-`as_of` bars |
| Purged CV split correctness | Synthetic signal set with known time-adjacency | Train/validation folds have the configured embargo gap; no adjacent-time leakage |
| Label computation matches hand-derivation | Synthetic price path, known candidate grid | Best-EV candidate pair matches a hand-computed value (same discipline as this repo's existing `BacktestEngine` tests) |
| Offline replay comparison | Run `kairos_signal_replay.py`-style replay with the ML wrapper vs. the static-percentile baseline over the same held-out window | Directional read on whether the ML brackets beat baseline EV/Sharaway — **not** a claim of live P&L, same caveat `kairos_signal_replay.py` already documents |
| Per-asset-class data sufficiency check | Count resolved, labeled signals per `asset_class_for()` bucket | Flag any class below `CLASS_STATS_MIN_SIGNALS` (30, existing repo constant) before training a per-class model on it |

## 7. Open risks / questions

- **Data imbalance across asset classes.** Per this project's own tracked state, the deduped
  signal corpus is ~95% equity (crypto n≈23, fx n≈6 groups, per prior session notes). A model
  trained pooled will mostly learn equity behavior; a model trained per-class will starve on
  crypto/fx. Needs a decision before training: pool with asset-class as a feature (risk: dominated
  by equity signal) vs. equity-only model for v1 (honest but narrower).
- **Overfitting**, explicitly warned about by the one rigorous source that covers this exact
  problem (AFML). Purged CV is necessary but not sufficient — held-out, out-of-time validation via
  §6's replay comparison is the real check.
- **Lookahead leakage** — this codebase has hit this bug shape twice already (see §4.2). Any
  implementer should re-read the "Naive-baseline zero-drift trap" and the 2026-09-01
  `_make_realized_predictions` sections of this repo's CLAUDE.md before writing feature-extraction
  code.
- **Label grid granularity** is an implementation choice not yet made (how fine a `(stop_pct,
  target_pct)` grid to search per historical signal) — coarser is cheaper but biases the label
  toward the grid's own resolution.

## 8. Implementation order

1. Label-generation script: reuse `kairos_signal_replay.py`'s `BacktestEngine._check_exit`/
   `_calculate_pnl` plumbing to compute the hindsight-best `(stop_pct, target_pct)` per historical
   resolved signal, over a small candidate grid. No model yet — just prove the label pipeline.
2. Feature extraction from already-computed `KairosDistribution` stats + ATR + `asset_class_for()`
   + horizon, with an explicit no-lookahead unit test (§6).
3. Train LightGBM quantile/regression model with purged CV; run the per-class data-sufficiency
   check (§6) before deciding pooled vs. per-class.
4. New wrapper strategy (`MLBracketStrategy` or similar), following the `ATRBracketStrategy`/
   `MetaLabelStrategy` shape.
5. Offline validation via a `kairos_signal_replay.py`-style comparison against the static-percentile
   baseline on a held-out window.
6. Only after 1-5 look promising: scope live-wiring as a separate, later design doc — same
   phasing discipline as the margin/leverage and offline-replay epics.
