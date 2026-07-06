# Kairos Framework: awesome-quant Gap Analysis & Design Document

**Source:** https://github.com/wilsonfreitas/awesome-quant (inventoried 2026-07-06)
**Status:** DESIGN — nothing in this document is implemented yet.

This document inventories every strategy type / quantitative technique referenced in
awesome-quant, maps it against the existing Kairos strategy suite, and specifies the
implementation of everything missing that fits Kairos's architecture (a Kronos
distribution-driven daily-bar backtest engine with `Strategy.generate_signal(dist,
current_price, history, context) -> Optional[Signal]`).

---

## 1. Inventory: awesome-quant vs. Kairos

### 1.1 Already covered (no work needed)

| awesome-quant technique | Kairos implementation |
|---|---|
| Mean reversion | `FadeExtremeStrategy`, `RangeTradingStrategy` |
| Momentum / trend following | `TrendFollowingStrategy`, `MomentumContinuationStrategy` |
| Pairs trading / cointegration | `CointegrationPairs`, `KalmanPairs`, `CopulaPairs`, `CointegrationECT` |
| Statistical arbitrage (cross-asset) | `CrossAssetSpreadStrategy`, `CrossAssetRankStrategy` |
| Cross-venue arbitrage | `ExchangeSpreadArbitrage`, `TriangularArbitrage` |
| Regime detection / switching | `HMMRegime`, `RegimeSwitchingStrategy`, `RegimeClusterStrategy`, `HurstRegimeSwitch` |
| Kelly sizing | `KairosDistribution.kelly_fraction`, used throughout |
| VaR position limits | `VaRPositionCapStrategy` |
| RSI / MACD / Bollinger | `RSIFilterStrategy`, `RSIDivergenceStrategy`, `MACDFilterStrategy`, `BollingerValidationStrategy` |
| VWAP execution benchmark | `PredictedVWAPStrategy` |
| Monte Carlo scenario analysis | `MonteCarloScenarioStrategy` |
| GNN inter-asset relationships | `GNNSectorRotation` |
| Reinforcement learning trading | `RLMetaController` |
| Transfer entropy / lead-lag | `TransferEntropy`, `MutualInformationWeight` |
| Online ensemble learning | `OnlineWeightedStrategy`, `ThompsonSamplingStrategy` |
| Gaussian process / Bayesian TS | `GaussianProcess`, `BSTSDecomposition`, `ParticleFilter` |
| Fractal / complexity measures | `FractalDimension`, `LZComplexity`, `RQADeterminism`, `DFAPersistence`, `WaveletMomentum` |
| Funding-rate / basis arb (crypto) | `FundingRateArbitrage`, `BasisTrade`, `FundingHarvest` |
| Liquidation cascade detection | `LiquidationFrontRun` |
| Carry trade (FX) | `CarryTrade` |
| Credit spread / CDS filters | `CDSSpreadFilter`, `OISSwapSpread` |
| M&A arbitrage | `MergerArb` |
| Earnings drift / catalyst | `PEAD`, `EarningsMomentum` |
| Insider trading signals | `InsiderCluster` |
| Sector rotation | `SectorRotation`, `SpectralClustering` |
| Transformer price forecasting | The Kronos model itself |
| Tail risk / options-like payoff shaping | `TailRiskStrategy`, `SellPremiumStrategy`, `BuyWingsStrategy`, `TailAsymmetryStrategy` |
| Pyramiding / time stops / overnight filters | `PyramidingStrategy`, `TimeBasedStopStrategy`, `OvernightExposureFilter` |
| Model decay monitoring | `ModelDecayMonitorStrategy` |
| Liquidity / volume confirmation | `LiquidityFilterStrategy`, `VolumeConfirmationStrategy`, `AmountFlowStrategy` |

### 1.2 Missing — in scope (specified in §2–§9)

Portfolio construction (MVO, risk parity, HRP/HERC, Black-Litterman, min-var +
shrinkage, eigen portfolios, universal portfolios, GA weights, CVaR optimization,
multi-asset Kelly), volatility modeling (GARCH, vol targeting, variance risk
premium), econometrics (ARIMA disagreement, VAR lead-lag, seasonality,
changepoint detection, Granger causality, matrix profile), ML (meta-labeling,
gradient-boosted direction classifier, LPPLS bubble detection), technical
indicators (Stochastic, ATR, OBV, ADX, multi-timeframe consensus), microstructure
proxies (volume profile, CVD approximation), sentiment/alt-data (news, social,
13F, economic calendar), execution (TWAP, implementation shortfall, TCA), factor
investing (cross-sectional multi-factor, PCA residual reversal), and framework
work (walk-forward validation, portfolio rebalancing engine).

### 1.3 Missing — explicitly out of scope (with rationale)

| Technique | Why not |
|---|---|
| Derivatives pricing (Black-Scholes, trees, LSMC, Heston/SABR, barrier/Asian, variance swaps, Greeks hedging) | Kairos has no options chain data feed; Kronos predicts spot OHLCV only. Revisit if an options data source is added. |
| XVA / counterparty risk | Institutional OTC concern; no counterparty exposure in this framework. |
| Fixed-income curve construction (Smith-Wilson, duration immunization, swaptions) | No bond/rates instrument support in the backtest engine. `CDSSpreadFilter`/`OISSwapSpread` already cover the tradable subset. |
| HFT / limit-order-book market making, footprint candles, whale order tracking | Engine operates on daily bars; queue-position simulation is a different engine entirely. |
| Prediction markets (Polymarket/Kalshi) | Different asset class and venue plumbing; not OHLCV. |
| DeFi (DEX routing, impermanent-loss hedging, MEV) | On-chain execution infrastructure, not a signal-generation problem. |
| AutoML hyperparameter search | Covered operationally by the pipeline's oracle stage; not a Strategy. |

---

## 2. New module: `strategy/kairos_portfolio.py`

Portfolio construction is a layer Kairos lacks entirely: everything today sizes
positions per-asset independently. These are **allocators**, not signal
strategies — they consume the per-asset signals/EVs produced each day and emit
target weights. Introduce one shared base:

```python
class PortfolioAllocator:
    name: str = "base_allocator"
    def allocate(self, signals: Dict[str, Signal], returns: pd.DataFrame,
                 dists: Dict[str, KairosDistribution], context: Dict) -> Dict[str, float]:
        """Return target weight per symbol (signed; sum of |w| <= 1)."""
        raise NotImplementedError
```

`returns` is a trailing window (default 120 days) of daily log returns for the
active universe. All allocators must handle: fewer than `min_obs` (default 60)
observations → fall back to equal weight among signaled assets; singular
covariance → apply shrinkage (§2.5) before solving. Dependencies: numpy/scipy
only (no cvxpy); solve QPs with `scipy.optimize.minimize(method="SLSQP")`.

### 2.1 Mean-Variance / Max-Sharpe — `mvo_allocator`
Markowitz with Kronos-derived expected returns: `mu[i] = dists[i].expected_value`
of the signal bracket (a genuinely forward-looking mu, unlike historical-mean
MVO). Maximize `(w·mu - rf) / sqrt(w'Σw)` subject to `sum|w| <= gross_cap`
(default 1.0), `|w_i| <= max_weight` (default 0.35), sign of `w_i` matching the
signal direction. Constructor: `__init__(lookback=120, gross_cap=1.0,
max_weight=0.35, rf=0.0)`.
**Acceptance:** with two uncorrelated assets of equal mu, weights split ~50/50;
raising one asset's mu monotonically raises its weight; never violates caps.

### 2.2 Risk Parity — `risk_parity_allocator`
Equal risk contribution: solve for `w` s.t. `w_i (Σw)_i` equal across signaled
assets (Spinu formulation, SLSQP on log-barrier objective). Direction from
signals, magnitude from ERC.
**Acceptance:** for 2 assets with vol 10%/20% and zero correlation, weights
≈ 2:1; risk contributions within 1% of each other at convergence.

### 2.3 Hierarchical Risk Parity / HERC — `hrp_allocator`
López de Prado HRP: correlation-distance matrix → `scipy.cluster.hierarchy.linkage`
(single) → quasi-diagonalization → recursive bisection with inverse-variance
splits. `variant="herc"` uses cluster-level equal risk contribution instead.
No matrix inversion — robust when n_assets approaches lookback.
**Acceptance:** reproduces the worked example from *Advances in Financial
Machine Learning* ch. 16 within 1e-6 on the same synthetic covariance; handles
n_assets=2 (degenerates to inverse-variance).

### 2.4 Black-Litterman — `black_litterman_allocator`
Prior: equilibrium returns `Π = δ Σ w_mkt` with `w_mkt` = inverse-vol weights
(no market-cap data needed). Views: one absolute view per signaled asset,
`Q_i = dists[i].stats["close"]["mean"]/price - 1`, with view uncertainty
`Ω_ii ∝ dists[i].entropy()` — high-entropy (uncertain) Kronos days get weak
views. Posterior mu feeds §2.1's optimizer. Constructor: `__init__(tau=0.05,
delta=2.5, lookback=120)`.
**Acceptance:** with zero-confidence views (Ω→∞) output equals prior weights;
with infinite-confidence views output matches MVO on Q; entropy=ln(20) view
moves posterior <10% of the way from prior to view.

### 2.5 Min-Variance + Ledoit-Wolf shrinkage — `minvar_allocator`
`sklearn.covariance.LedoitWolf` if sklearn present, else manual shrinkage to
scaled identity with the Ledoit-Wolf closed-form intensity. Minimize `w'Σw`
under the same caps as §2.1. The shrinkage helper `shrunk_covariance(returns)`
is module-level and reused by §2.1–§2.4.
**Acceptance:** shrinkage intensity in [0,1]; with n=200 obs / 3 assets,
intensity < 0.3; output covariance is positive definite for n_assets > n_obs.

### 2.6 Eigen Portfolios — `eigen_allocator`
PCA on the correlation matrix; drop PC1 (market mode); allocate to the top-k
remaining eigenvectors weighted by eigenvalue, projected back to asset space and
re-signed by signal direction. Constructor: `__init__(n_components=3, lookback=120)`.
**Acceptance:** eigen portfolios are mutually orthogonal; PC1 exclusion reduces
average pairwise correlation of resulting weight vector with equal-weight basket.

### 2.7 Universal Portfolio (Cover) — `universal_allocator`
Online learning: maintain a grid of constant-rebalanced portfolios over the
signaled assets (Dirichlet grid, resolution 0.1), track each grid point's
cumulative wealth, output the wealth-weighted mixture. State persists across
days in the allocator instance (same pattern as `StrategyPerformanceTracker`).
**Acceptance:** on synthetic data where one asset dominates, weights converge
toward it; total weight always sums to 1; grid regenerated when universe changes.

### 2.8 Genetic Algorithm Weights — `ga_allocator`
Fitness = trailing 60-day Sharpe of the weight vector applied to `returns`;
population 50, tournament selection, blend crossover, Gaussian mutation σ=0.05,
20 generations, re-run weekly (cached otherwise). Deterministic seed from date.
**Acceptance:** fitness non-decreasing across generations on fixed data; weekly
cache verified (identical output within the week); respects §2.1 caps.

### 2.9 CVaR Optimization — `cvar_allocator`
Rockafellar-Uryasev LP formulation over **Kronos sample paths**: use the
`PRED_SAMPLES` predicted returns per asset as the scenario set (not historical
returns — this is the differentiator), minimize CVaR_95 subject to
`w·mu >= target_return`. Solve with `scipy.optimize.linprog`.
**Acceptance:** CVaR of chosen weights <= CVaR of equal weight on the same
scenario set; infeasible target_return falls back to max-return vertex.

### 2.10 Multi-Asset Kelly — `kelly_allocator`
`w = f · Σ⁻¹ mu` (continuous-time Kelly) with `mu` from Kronos EVs, Σ shrunk
(§2.5), fractional-Kelly `f=0.25` default, clipped to caps.
**Acceptance:** single-asset case reduces to `kelly_fraction` within tolerance;
doubling Σ halves weights.

---

## 3. New module: `strategy/kairos_volatility.py`

### 3.1 GARCH(1,1) Volatility Filter — `garch_filter`
**Type:** filter wrapper. Fit GARCH(1,1) by MLE on trailing 250 log returns
(scipy, no `arch` dependency; 3 params, L-BFGS-B, variance targeting for ω).
Forecast next-day σ. Block the wrapped strategy's signal when forecast σ exceeds
`sigma_cap` (default: 90th percentile of trailing fitted σ) — the volatility
analogue of `KurtosisFilterStrategy`. Refit weekly, cached.
**Acceptance:** on simulated GARCH data recovers α+β within ±0.1; blocks during
simulated vol spikes; falls back to pass-through (with warning in context) if
MLE fails to converge.

### 3.2 Volatility Targeting Sizer — `vol_target_sizer`
**Type:** sizing wrapper. Scale `signal.size` by `target_vol / blended_vol`
where blended_vol = 0.5·(GARCH forecast) + 0.5·(Kronos predicted range vol =
`(pct_84 - pct_16) / (2·price)`). `target_vol` default 15% annualized.
**Acceptance:** size halves when blended vol doubles; never exceeds base size
× `max_leverage` (default 2.0); never increases a zero-size signal.

### 3.3 Variance Risk Premium — `variance_risk_premium`
**Type:** standalone strategy. Kronos-implied variance (from prediction sample
dispersion) vs. trailing 20-day realized variance. When implied >> realized
(ratio > `entry_ratio`, default 1.5) the model expects a vol expansion — take
the straddle-proxy: enter in the direction of `TailAsymmetryStrategy`-style
skew with wide bracket (stop pct_5/target pct_95). When implied << realized,
expect compression — `RangeTradingStrategy`-style fade with tight bracket.
**Acceptance:** no signal when ratio in [1/entry_ratio, entry_ratio]; bracket
widths verified against distribution percentiles.

### 3.4 ATR Stop/Sizing — `atr_bracket`
**Type:** bracket modifier wrapper. Recompute wrapped signal's stop at
`entry ∓ k_stop·ATR(14)` and target at `entry ± k_target·ATR(14)` (Wilder
smoothing on `history`), keeping the tighter of {ATR stop, original stop}.
Defaults k_stop=2.0, k_target=3.0.
**Acceptance:** ATR matches TA-Lib reference values on a fixture to 1e-6;
stop only ever tightens; direction-consistent (stop below entry for longs).

---

## 4. New module: `strategy/kairos_econometric.py`

### 4.1 ARIMA Disagreement Filter — `arima_disagreement`
**Type:** filter wrapper. Fit AR(p) with drift (statsmodels-free: OLS on lags,
p selected by AIC over 1..5) on trailing 120 closes. If the ARIMA point
forecast and the Kronos mean forecast disagree in direction, veto the signal;
if they agree, boost `confidence` by `agree_boost` (default 1.2, capped at 1.0).
**Acceptance:** veto fires iff sign mismatch; on a pure trend series AR forecast
sign matches trend direction.

### 4.2 VAR Lead-Lag — `var_leadlag`
**Type:** standalone strategy. Fit VAR(1) via OLS on the 3-asset return panel.
If asset j's lagged return significantly (|t|>2) predicts asset i's return, and
yesterday's j-move implies an i-move that *agrees* with Kronos direction for i,
emit signal on i with `DynamicBracketStrategy`-style bracket. Complements
`TransferEntropy` (linear vs. nonlinear lead-lag).
**Acceptance:** on synthetic data with planted x→y lag-1 dependence, detects the
edge and only that edge; no signal when coefficient insignificant.

### 4.3 Calendar Seasonality — `seasonality_filter`
**Type:** filter wrapper. STL-lite decomposition: day-of-week and month-of-year
mean effects estimated on trailing 2 years with HAC-adjusted t-stats. Veto
signals that fight a significant (|t|>2) seasonal effect; pass otherwise. Uses
`kairos.calendar` for trading-day awareness.
**Acceptance:** detects a planted Friday effect in synthetic data; no vetoes
when all effects insignificant (t below threshold on white noise, 95% of runs).

### 4.4 Changepoint Detection — `changepoint_guard`
**Type:** filter wrapper. Bayesian online changepoint detection (Adams &
MacKay, hazard 1/60, Normal-Inverse-Gamma conjugate) on daily returns. When
P(run length < 5) > 0.5 — a fresh regime break — veto all signals for
`cooloff_days` (default 3): the Kronos context window is contaminated by the
old regime. This is a distinct mechanism from `HMMRegime` (which *selects*
between known regimes rather than detecting novel breaks).
**Acceptance:** on synthetic mean-shift series, detects break within 3 days at
<5% false-positive rate on white noise; cooloff countdown verified.

### 4.5 Granger Causality Pairs — `granger_pairs`
**Type:** standalone strategy. Rolling Granger F-test (lags 1-3) between all
asset pairs; trade the follower in the direction implied by the leader's
yesterday move × the fitted coefficient sign, gated on Kronos agreement.
Shares the OLS machinery of §4.2 (module-level `_lagged_ols` helper).
**Acceptance:** F-test p-values match statsmodels reference on a fixture within
1e-4; symmetric independence produces no signals.

### 4.6 Matrix Profile Anomaly — `matrix_profile_anomaly`
**Type:** standalone strategy. STOMP over trailing 250 closes (window 20,
z-normalized, pure numpy). If today's window is a **discord** (profile value >
mean + 2σ): abstain — unprecedented pattern, model unreliable. If a strong
**motif** match exists: trade the direction that followed the historical match,
gated on Kronos agreement, sized by match quality.
**Acceptance:** matrix profile matches the `stumpy` reference implementation on
a fixture within 1e-4; planted repeated motif is found; discord abstention fires
on a planted anomaly.

---

## 5. New module: `strategy/kairos_ml.py`

### 5.1 Meta-Labeling — `meta_label`
**Type:** filter wrapper (the flagship item in this module). López de Prado
meta-labeling on top of any base strategy: label each historical base-strategy
signal by triple-barrier outcome (profit-take/stop/time from the signal's own
bracket), train a secondary classifier P(signal wins) on features {entropy,
kurtosis, skew, CDF position, ATR ratio, trailing strategy hit-rate, regime
id}, and size the live signal by the predicted probability (veto below
`p_min=0.55`). Classifier: logistic regression (numpy IRLS, no sklearn dep);
warm-up 60 labeled signals before it activates (pass-through until then).
Persists labeled history in the wrapper instance.
**Acceptance:** on a synthetic setup where signals win iff entropy < 2.0, the
trained model achieves AUC > 0.9 and vetoes high-entropy signals; pass-through
verified during warm-up.

### 5.2 Gradient-Boosted Direction Classifier — `gbm_direction`
**Type:** standalone strategy. Small gradient-boosted stumps (numpy
implementation: 50 trees, depth 2, lr 0.1, logloss) predicting next-day
direction from ~15 technical features (returns at 1/5/20d, RSI, ATR ratio,
volume z-score, day-of-week, Kronos summary stats). Trades only when classifier
and Kronos agree AND P > 0.6. Retrain weekly on trailing 500 days.
**Acceptance:** beats logistic baseline on synthetic nonlinear (XOR-of-features)
data; deterministic given seed; retrain cadence verified.

### 5.3 LPPLS Bubble Detection — `lppls_guard`
**Type:** filter wrapper. Fit the log-periodic power law singularity model
(Sornette) on trailing 250 log-prices: nonlinear fit of (tc, m, ω) with the
linear params profiled out (standard 3-param reduction), multi-start
Nelder-Mead. When fit quality passes the Sornette filter conditions
(0.1 < m < 0.9, 6 < ω < 13, tc within 60 days) — a bubble signature — veto new
LONG entries and allow/boost SHORT signals.
**Acceptance:** flags a synthetic super-exponential + log-periodic series;
does not flag GBM paths (false-positive rate < 10% over 100 seeds).

---

## 6. Additions to existing modules

### 6.1 Technical filters → `kairos_backtest.py` (alongside RSI/MACD/Bollinger)

| ID | Spec | Acceptance |
|---|---|---|
| `stochastic_filter` | %K(14)/%D(3) filter wrapper: veto longs when %K>80 unless trending (ADX>25), veto shorts when %K<20. | %K/%D match TA-Lib fixture; veto logic truth-table tested. |
| `adx_gate` | Wrapper routing by trend strength: ADX(14)>25 passes trend-type strategies, ADX<20 passes mean-reversion-type; strategy type declared at wrap time (`kind="trend"\|"reversion"`). | ADX matches fixture; routing verified both ways. |
| `obv_confirmation` | OBV(20-slope) must agree with signal direction, else veto. Complements `VolumeConfirmationStrategy` (which uses predicted volume; OBV uses realized). | Slope-sign agreement logic tested; flat OBV passes through. |
| `mtf_consensus` | Multi-timeframe consensus: resample history to {1d, 3d, 1w}, compute trend sign (EMA20 vs EMA50 equivalent per frame); require ≥2/3 agreement with signal direction. | Resampling boundary-correct via `kairos.calendar`; 2/3 vote logic tested. |

All four are pure-history wrappers following the exact `RSIFilterStrategy`
pattern (return `Signal` dataclass or `None` — never a dict; see CLAUDE.md).

### 6.2 Microstructure proxies → `kairos_execution.py`

| ID | Spec | Acceptance |
|---|---|---|
| `volume_profile_levels` | Build a 60-day volume-at-price histogram (20 bins between rolling min/max); snap wrapped signal's stop/target to nearest high-volume node (support) / low-volume node (target through the gap). Complements `SupportConfluenceStrategy`. | POC/VAH/VAL computed correctly on fixture; stop only moves to a *nearer* level. |
| `cvd_divergence` | Cumulative volume delta approximation from daily bars (volume signed by close-vs-open); trade divergence between 20-day CVD slope and price slope, gated on Kronos agreement. | Sign convention tested; divergence detection on planted fixture. |

True order-book imbalance is listed out-of-scope (§1.3) until intraday L2 data exists.

### 6.3 Execution algorithms → `kairos_execution.py`

| ID | Spec | Acceptance |
|---|---|---|
| `twap_execution` | Wrapper: split entry across the first k in-day steps of the predicted path (reuses `PathExecutionStrategy` plumbing), recording per-slice fills in `signal.metadata["fills"]`; average fill becomes effective entry. | Effective entry = mean of slice prices; bracket recomputed off effective entry. |
| `implementation_shortfall` | Wrapper: choose between immediate-fill and TWAP per signal by comparing Kronos-predicted drift over the execution horizon vs. assumed impact cost (`impact_bps` param, default 5): fast when drift adverse, patient when favorable. | Decision boundary tested at drift = impact; both branches exercised. |
| `tca_report` | Not a strategy: post-backtest function `compute_tca(trades) -> DataFrame` decomposing per-trade slippage into timing (entry vs. day open) and impact (assumed bps) components; wire into `compute_metrics` output as optional section. | Columns sum to total slippage per trade; empty trade list handled. |

### 6.4 Factor investing → `kairos_meta.py`

| ID | Spec | Acceptance |
|---|---|---|
| `multi_factor_rank` | Cross-sectional composite over the universe: momentum (12-1 return), low-vol (inverse 60d σ), "value" proxy (distance below 252d high), quality proxy (return/σ stability). Z-score each, average, long top decile / short bottom (or top-1/bottom-1 for 3-asset mode), gated on Kronos agreement. Extends `CrossAssetRankStrategy` from 1 factor to k. | Z-scoring and composite ranks verified on fixture; degenerates to `CrossAssetRankStrategy` behavior with momentum-only weights. |
| `pca_residual_reversal` | Statistical factor model: PCA (k=1) on 60d returns, compute each asset's residual vs. factor reconstruction; fade assets with |residual z| > 2 back toward the factor, gated on Kronos agreement. | Residuals orthogonal to factor; reversal fires on planted idiosyncratic shock. |

### 6.5 Sentiment / alternative data → new module `strategy/kairos_sentiment.py`

These follow the **context graceful degradation** pattern established in
`EXTENDED_STRATEGIES_DONE.md`: each reads a `context` key and returns `None`
(pass-through for wrappers) when the key is absent, so they are safe to
register before any data feed exists. Data ingestion itself is a separate
pipeline task (out of this doc's scope beyond the key contract).

| ID | Context key contract | Signal logic |
|---|---|---|
| `news_sentiment_filter` | `context["news_sentiment"][symbol]` ∈ [-1,1] | Veto signals fighting strong opposing sentiment (|s| > 0.5); boost confidence when aligned. |
| `social_momentum` | `context["social_mentions"][symbol]` = {count, z_score, sentiment} | Standalone: mention z>3 with positive sentiment → momentum long (crowd inflow); z>3 with price already +20%/5d → fade (blow-off proxy). Gated on Kronos. |
| `institutional_13f` | `context["inst_ownership_delta"][symbol]` ∈ quarterly Δ% | Filter: veto shorts against strong institutional accumulation (Δ > +2%); complements `InsiderCluster` and `DarkPoolFilter`. |
| `econ_calendar_guard` | `context["econ_events"]` = list of {date, impact} | Veto new entries the day before high-impact events (CPI/NFP/FOMC); tighten stops on open positions via metadata flag. |

**Acceptance (all four):** missing context key → identical behavior to
unwrapped strategy (regression-tested); each veto/boost path unit-tested with
synthetic context.

---

## 7. Framework work (not strategies)

### 7.1 Walk-Forward Validation — `kairos_backtest.py`
`walk_forward(strategy_factory, data, train_days=250, test_days=60, step=60)`:
rolls anchored or sliding windows, calls the factory fresh per fold (no state
leakage), runs `backtest()` per test window, returns per-fold and aggregate
`compute_metrics()` plus an overfitting score (in-sample vs. out-of-sample
Sharpe ratio degradation, and Deflated Sharpe per Bailey & López de Prado).
**Acceptance:** folds never overlap in test data; a deliberately overfit
strategy (lookahead-peeking fixture) shows OOS Sharpe collapse and DSR < 0.5;
a fixed-seed run is reproducible.

### 7.2 Rebalancing Engine — `kairos_portfolio.py`
`Rebalancer(allocator, mode="threshold", band=0.05, min_interval_days=5)`:
converts allocator target weights (§2) into trade deltas only when drift
exceeds `band` or `mode="periodic"` interval elapses; respects per-trade
transaction cost from the backtest config so turnover is penalized.
**Acceptance:** no trades within band; turnover under threshold mode < turnover
under daily full rebalance on the same weight stream.

### 7.3 Orchestrator integration
- `StrategyRegistry` (kairos_orchestrator.py:270) gains a `register_allocator()`
  path; the orchestrator applies the active allocator **after** per-asset signal
  generation and meta-filters, replacing per-signal `size` with allocator weight
  (per-signal size becomes the within-asset cap).
- New context keys: `context["returns_window"]` (trailing return panel, computed
  once per day for all §2/§4/§6.4 strategies), `context["realized_vol"]`.
- All new wrappers must be registrable under the existing per-class
  disabled-strategy fallback added in f0662fd.

---

## 8. Cross-cutting implementation rules

1. Every `generate_signal` returns `Signal` or `None` — never a dict
   (LiquidityFilterStrategy wrapper breaks otherwise; see CLAUDE.md).
2. Percentile keys are `f"pct_{int(x)}"` — cast floats.
3. No new hard dependencies: numpy/scipy/pandas only. sklearn is optional
   (§2.5) with a manual fallback. No `arch`, `statsmodels`, `cvxpy`, `stumpy` —
   each has a small pure-numpy implementation specified above, with reference
   fixtures generated once from the real library and checked in.
4. Anything that fits a model (GARCH, GBM, LPPLS, meta-labeler, GA) caches and
   refits on a fixed cadence — never per-bar — to protect the 0.42 s/iter GPU
   backtest budget. Fitting must be CPU-only and off the inference hot path.
5. Stateful strategies (universal portfolio, meta-labeler, changepoint) keep
   state on the instance, following `StrategyPerformanceTracker`'s pattern, and
   must expose `reset()` for walk-forward folds.
6. Entropy/kurtosis thresholds: reuse `OrchestratorConfig` values; do not
   introduce parallel thresholds (§5.1 features consume the existing ones).

---

## 9. Implementation order

| Phase | Items | Rationale |
|---|---|---|
| 1 | §7.1 walk-forward, §2 base class + shrinkage helper (§2.5), §3.4 ATR | Validation harness first — everything after gets evaluated through it; ATR is a dependency of §3.2/§5.2/§6.1. |
| 2 | §2.1–§2.3, §2.10, §7.2, §7.3 | Core portfolio layer + orchestrator wiring: biggest structural gap. |
| 3 | §3.1–§3.3, §4.4 | Vol modeling + changepoint: highest expected risk-adjusted impact filters. |
| 4 | §5.1 meta-labeling, §6.1 technical filters | Meta-labeling is the highest-value single strategy in the doc; technical filters are cheap wins. |
| 5 | §4.1–§4.3, §4.5–§4.6, §6.4 | Econometrics + factors. |
| 6 | §2.4, §2.6–§2.9, §5.2–§5.3, §6.2–§6.3 | Long tail: BL, eigen/universal/GA/CVaR allocators, GBM, LPPLS, microstructure, execution. |
| 7 | §6.5 sentiment scaffolding | Key-contract stubs last; real value blocked on data feeds. |

Per-phase definition of done: unit tests in `tests/unit/` (no GPU, fixtures
checked in), registered in `StrategyRegistry`, one walk-forward run on the
3-asset demo data documented in the module docstring, entry added to
`EXTENDED_STRATEGIES_DONE.md`.

## 10. Testing plan

- New test files mirror existing convention: `tests/unit/test_portfolio.py`,
  `test_volatility.py`, `test_econometric.py`, `test_ml_strategies.py`,
  `test_sentiment.py`, `test_execution_algos.py`, `test_walk_forward.py`.
- Reference fixtures (TA-Lib values, statsmodels Granger p-values, stumpy
  matrix profiles, AFML ch.16 HRP example) generated by a one-off script in
  `scripts/gen_reference_fixtures.py` and committed as CSV — the libraries are
  dev-time-only and never imported by strategy code.
- Every wrapper gets the standard three: pass-through on None base signal,
  never-widens/never-increases invariant, missing-context degradation.
- `tests/conftest.py` already puts `strategy/` on `sys.path`; no changes needed.
