# Phase 4 — Paper trading & performance accounting

**Goal:** prove the live advice stream matches backtest expectations before any
real money. Do not skip this phase.

**Depends on:** Phase 3.
**Rough effort:** 2-3 subagent-days (Sonnet 5), then 4-6 weeks of calendar time
collecting data.

## Tasks

### 4.1 Paper-trade executor
- Every recommendation is "executed" at next-bar open in `data/positions.db`;
  slippage assumptions and fees per asset class are applied.
- Nightly reconciliation computes realized P&L, Sharpe of the **live advice
  stream**, and per-strategy attribution.
- Owner: 1 Sonnet subagent.

### 4.2 Live vs backtest drift monitor
- Weekly job comparing live-signal performance against the backtest expectation
  for the same profile.
- Telegram alert when a strategy underperforms its backtest band — this is the
  trigger to re-run the oracle stage and refresh disabled lists.
- Owner: 1 Sonnet subagent.

### 4.3 Profile refresh loop
- Documented (later scheduled, e.g. monthly) re-run of pipeline stages 1-4,
  diffing newly suggested profiles against `config/profiles.yaml`.
- Owner: orchestrator + Haiku for the diff script.

## Exit criteria
At least 4-6 weeks of paper trading; live advice Sharpe consistent with
backtest; drift monitor quiet.
