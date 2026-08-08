# Kairos Awesome-Quant Gap Implementation — Todo

## Epic E1 — Foundation

- [x] E1-S01 Walk-forward Harness (E1-S01-walk-forward-harness.md)
- [x] E1-S02 Allocator Base and Shrinkage (E1-S02-allocator-base-and-shrinkage.md)
- [x] E1-S03 ATR Bracket Wrapper (E1-S03-atr-bracket-wrapper.md)

## Epic E2 — Portfolio Allocators

- [x] E2-S01 MVO Allocator (E2-S01-mvo-allocator.md)
- [x] E2-S02 Risk Parity Allocator (E2-S02-risk-parity-allocator.md)
- [x] E2-S03 HRP Allocator (E2-S03-hrp-allocator.md)
- [x] E2-S04 MinVar Allocator (E2-S04-minvar-allocator.md)
- [x] E2-S05 Black-Litterman Allocator (E2-S05-black-litterman-allocator.md)
- [x] E2-S06 Eigen Allocator (E2-S06-eigen-allocator.md)
- [x] E2-S07 Universal Allocator (E2-S07-universal-allocator.md)
- [x] E2-S08 GA Allocator (E2-S08-ga-allocator.md)
- [x] E2-S09 CVaR Allocator (E2-S09-cvar-allocator.md)
- [x] E2-S10 Kelly Allocator (E2-S10-kelly-allocator.md)
- [x] E2-S11 Rebalancer (E2-S11-rebalancer.md)
- [x] E2-S12 Orchestrator Allocator Integration (E2-S12-orchestrator-allocator-integration.md)

## Epic E3 — Volatility

- [x] E3-S01 GARCH Filter (E3-S01-garch-filter.md)
- [x] E3-S02 Vol Target Sizer (E3-S02-vol-target-sizer.md)
- [x] E3-S03 Variance Risk Premium (E3-S03-variance-risk-premium.md)

## Epic E4 — Econometrics

- [x] E4-S01 Lagged OLS and ARIMA Disagreement (E4-S01-lagged-ols-and-arima-disagreement.md)
- [x] E4-S02 VAR Lead-Lag (E4-S02-var-leadlag.md)
- [x] E4-S03 Seasonality Filter (E4-S03-seasonality-filter.md)
- [x] E4-S04 Changepoint Guard (E4-S04-changepoint-guard.md)
- [x] E4-S05 Granger Pairs (E4-S05-granger-pairs.md)
- [x] E4-S06 Matrix Profile Anomaly (E4-S06-matrix-profile-anomaly.md)

## Epic E5 — ML

- [x] E5-S01 Meta-Labeling (E5-S01-meta-labeling.md)
- [x] E5-S02 GBM Direction (E5-S02-gbm-direction.md)
- [x] E5-S03 LPPLS Guard (E5-S03-lppls-guard.md)

## Epic E6 — Technical Filters

- [x] E6-S01 Stochastic Filter (E6-S01-stochastic-filter.md)
- [x] E6-S02 ADX Gate (E6-S02-adx-gate.md)
- [x] E6-S03 OBV Confirmation (E6-S03-obv-confirmation.md)
- [x] E6-S04 MTF Consensus (E6-S04-mtf-consensus.md)

## Epic E7 — Execution & Microstructure

- [x] E7-S01 Volume Profile Levels (E7-S01-volume-profile-levels.md)
- [x] E7-S02 CVD Divergence (E7-S02-cvd-divergence.md)
- [x] E7-S03 TWAP Execution (E7-S03-twap-execution.md)
- [x] E7-S04 Implementation Shortfall (E7-S04-implementation-shortfall.md)
- [x] E7-S05 TCA Report (E7-S05-tca-report.md)

## Epic E8 — Factors

- [x] E8-S01 Multi-Factor Rank (E8-S01-multi-factor-rank.md)
- [x] E8-S02 PCA Residual Reversal (E8-S02-pca-residual-reversal.md)

## Epic E9 — Sentiment Scaffolding

- [x] E9-S01 News Sentiment Filter (E9-S01-news-sentiment-filter.md)
- [x] E9-S02 Social Momentum (E9-S02-social-momentum.md)
- [x] E9-S03 Institutional 13F (E9-S03-institutional-13f.md)
- [x] E9-S04 Econ Calendar Guard (E9-S04-econ-calendar-guard.md)

## Epic E10 — Pipeline Automation (stages 1-4 + viability report)

- [x] E10-S01 Period-to-weeks Helper (E10-S01-period-to-weeks.md)
- [x] E10-S02 Viability Report Builder (E10-S02-viability-report.md)
- [x] E10-S03 run_stage_auto Chaining (E10-S03-run-stage-auto.md)
- [x] E10-S04 CLI Wiring (E10-S04-cli-wiring.md)
- [x] E10-S05 Interval-aware Correlation (E10-S05-correlation-interval.md)
- [x] E10-S06 PIPELINE.md Docs (E10-S06-pipeline-docs.md)

## Epic E11 — Portfolio Allocation Sheet

- [x] E11-S01 Candidate Schema + fetch_signals (E11-S01-candidate-schema-fetch.md)
- [x] E11-S02 SCHEMA_ERROR Validation (E11-S02-schema-validation.md)
- [x] E11-S03 Config + Per-row Derived Columns (E11-S03-config-and-derived-columns.md)
- [x] E11-S04 ev_implied Data Quality Check (E11-S04-data-quality-check.md)
- [x] E11-S05 Selection: Gate, Collapse, Top-K (E11-S05-selection-gate-collapse-topk.md)
- [x] E11-S06 Sizing: Kelly Cap, Cluster Caps, Dust (E11-S06-sizing-caps-dust.md)
- [x] E11-S07 allocate() Orchestration + Cluster Map (E11-S07-allocate-orchestration.md)
- [x] E11-S08 Formula Template Engine (E11-S08-formula-template-engine.md)
- [x] E11-S09 XLSX Sheet Writer (E11-S09-xlsx-sheet-writer.md)
- [x] E11-S10 ODS Sheet Writer (E11-S10-ods-sheet-writer.md)
- [x] E11-S11 Markdown Section Writer (E11-S11-markdown-section-writer.md)
- [x] E11-S12 Wire into kairos_signals.py (E11-S12-wire-into-kairos-signals.md)
- [x] E11-S13 LibreOffice Parity Tests (E11-S13-libreoffice-parity-tests.md)
- [x] E11-S14 Golden-file + Property Tests (E11-S14-remaining-unit-tests.md)
