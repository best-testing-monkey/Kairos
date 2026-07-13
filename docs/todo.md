# Kairos Awesome-Quant Gap Implementation — Todo

## Epic E1 — Foundation

- [x] E1-S01 Walk-forward Harness (docs/tickets/E1-S01-walk-forward-harness.md)
- [x] E1-S02 Allocator Base and Shrinkage (docs/tickets/E1-S02-allocator-base-and-shrinkage.md)
- [x] E1-S03 ATR Bracket Wrapper (docs/tickets/E1-S03-atr-bracket-wrapper.md)

## Epic E2 — Portfolio Allocators

- [x] E2-S01 MVO Allocator (docs/tickets/E2-S01-mvo-allocator.md)
- [x] E2-S02 Risk Parity Allocator (docs/tickets/E2-S02-risk-parity-allocator.md)
- [x] E2-S03 HRP Allocator (docs/tickets/E2-S03-hrp-allocator.md)
- [x] E2-S04 MinVar Allocator (docs/tickets/E2-S04-minvar-allocator.md)
- [x] E2-S05 Black-Litterman Allocator (docs/tickets/E2-S05-black-litterman-allocator.md)
- [x] E2-S06 Eigen Allocator (docs/tickets/E2-S06-eigen-allocator.md)
- [x] E2-S07 Universal Allocator (docs/tickets/E2-S07-universal-allocator.md)
- [x] E2-S08 GA Allocator (docs/tickets/E2-S08-ga-allocator.md)
- [x] E2-S09 CVaR Allocator (docs/tickets/E2-S09-cvar-allocator.md)
- [x] E2-S10 Kelly Allocator (docs/tickets/E2-S10-kelly-allocator.md)
- [x] E2-S11 Rebalancer (docs/tickets/E2-S11-rebalancer.md)
- [x] E2-S12 Orchestrator Allocator Integration (docs/tickets/E2-S12-orchestrator-allocator-integration.md)

## Epic E3 — Volatility

- [x] E3-S01 GARCH Filter (docs/tickets/E3-S01-garch-filter.md)
- [x] E3-S02 Vol Target Sizer (docs/tickets/E3-S02-vol-target-sizer.md)
- [x] E3-S03 Variance Risk Premium (docs/tickets/E3-S03-variance-risk-premium.md)

## Epic E4 — Econometrics

- [x] E4-S01 Lagged OLS and ARIMA Disagreement (docs/tickets/E4-S01-lagged-ols-and-arima-disagreement.md)
- [x] E4-S02 VAR Lead-Lag (docs/tickets/E4-S02-var-leadlag.md)
- [x] E4-S03 Seasonality Filter (docs/tickets/E4-S03-seasonality-filter.md)
- [x] E4-S04 Changepoint Guard (docs/tickets/E4-S04-changepoint-guard.md)
- [x] E4-S05 Granger Pairs (docs/tickets/E4-S05-granger-pairs.md)
- [x] E4-S06 Matrix Profile Anomaly (docs/tickets/E4-S06-matrix-profile-anomaly.md)

## Epic E5 — ML

- [x] E5-S01 Meta-Labeling (docs/tickets/E5-S01-meta-labeling.md)
- [x] E5-S02 GBM Direction (docs/tickets/E5-S02-gbm-direction.md)
- [x] E5-S03 LPPLS Guard (docs/tickets/E5-S03-lppls-guard.md)

## Epic E6 — Technical Filters

- [x] E6-S01 Stochastic Filter (docs/tickets/E6-S01-stochastic-filter.md)
- [x] E6-S02 ADX Gate (docs/tickets/E6-S02-adx-gate.md)
- [x] E6-S03 OBV Confirmation (docs/tickets/E6-S03-obv-confirmation.md)
- [x] E6-S04 MTF Consensus (docs/tickets/E6-S04-mtf-consensus.md)

## Epic E7 — Execution & Microstructure

- [x] E7-S01 Volume Profile Levels (docs/tickets/E7-S01-volume-profile-levels.md)
- [x] E7-S02 CVD Divergence (docs/tickets/E7-S02-cvd-divergence.md)
- [x] E7-S03 TWAP Execution (docs/tickets/E7-S03-twap-execution.md)
- [x] E7-S04 Implementation Shortfall (docs/tickets/E7-S04-implementation-shortfall.md)
- [x] E7-S05 TCA Report (docs/tickets/E7-S05-tca-report.md)

## Epic E8 — Factors

- [x] E8-S01 Multi-Factor Rank (docs/tickets/E8-S01-multi-factor-rank.md)
- [x] E8-S02 PCA Residual Reversal (docs/tickets/E8-S02-pca-residual-reversal.md)

## Epic E9 — Sentiment Scaffolding

- [x] E9-S01 News Sentiment Filter (docs/tickets/E9-S01-news-sentiment-filter.md)
- [x] E9-S02 Social Momentum (docs/tickets/E9-S02-social-momentum.md)
- [x] E9-S03 Institutional 13F (docs/tickets/E9-S03-institutional-13f.md)
- [x] E9-S04 Econ Calendar Guard (docs/tickets/E9-S04-econ-calendar-guard.md)

## Epic E10 — Pipeline Automation (stages 1-4 + viability report)

- [x] E10-S01 Period-to-weeks Helper (docs/tickets/E10-S01-period-to-weeks.md)
- [x] E10-S02 Viability Report Builder (docs/tickets/E10-S02-viability-report.md)
- [x] E10-S03 run_stage_auto Chaining (docs/tickets/E10-S03-run-stage-auto.md)
- [x] E10-S04 CLI Wiring (docs/tickets/E10-S04-cli-wiring.md)
- [x] E10-S05 Interval-aware Correlation (docs/tickets/E10-S05-correlation-interval.md)
- [x] E10-S06 PIPELINE.md Docs (docs/tickets/E10-S06-pipeline-docs.md)

## Epic E11 — Portfolio Allocation Sheet

- [x] E11-S01 Candidate Schema + fetch_signals (docs/tickets/E11-S01-candidate-schema-fetch.md)
- [ ] E11-S02 SCHEMA_ERROR Validation (docs/tickets/E11-S02-schema-validation.md)
- [ ] E11-S03 Config + Per-row Derived Columns (docs/tickets/E11-S03-config-and-derived-columns.md)
- [ ] E11-S04 ev_implied Data Quality Check (docs/tickets/E11-S04-data-quality-check.md)
- [ ] E11-S05 Selection: Gate, Collapse, Top-K (docs/tickets/E11-S05-selection-gate-collapse-topk.md)
- [ ] E11-S06 Sizing: Kelly Cap, Cluster Caps, Dust (docs/tickets/E11-S06-sizing-caps-dust.md)
- [ ] E11-S07 allocate() Orchestration + Cluster Map (docs/tickets/E11-S07-allocate-orchestration.md)
- [ ] E11-S08 Formula Template Engine (docs/tickets/E11-S08-formula-template-engine.md)
- [ ] E11-S09 XLSX Sheet Writer (docs/tickets/E11-S09-xlsx-sheet-writer.md)
- [ ] E11-S10 ODS Sheet Writer (docs/tickets/E11-S10-ods-sheet-writer.md)
- [ ] E11-S11 Markdown Section Writer (docs/tickets/E11-S11-markdown-section-writer.md)
- [ ] E11-S12 Wire into kairos_signals.py (docs/tickets/E11-S12-wire-into-kairos-signals.md)
- [ ] E11-S13 LibreOffice Parity Tests (docs/tickets/E11-S13-libreoffice-parity-tests.md)
- [ ] E11-S14 Golden-file + Property Tests (docs/tickets/E11-S14-remaining-unit-tests.md)
