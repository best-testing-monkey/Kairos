# Factor 2: Portfolio-level risk aggregation / concurrent exposure

Source: `docs/papertrade_loss_analysis.md` §4, Factor 2

## Problem

The day-loop in `kairos_papertrade.py:504-553` sizes each day's *new* orders
against available cash with `top_k=args.top_n` (default 3) and `max_pos_pct=15`,
but nothing caps how many *previously opened, still-open* positions can be
outstanding at once. Positions accumulate across days until they individually
hit SL/TP. In the recorded 2026-07-23 run, account cash fell as low as **€74.22
(37% of capital)** and cumulative traded notional over the window was **€9,665
— 48x capital turnover**.

This is flagged as the single biggest lever behind a positive per-trade mean
turning into a negative/volatile total return: high concurrent exposure
amplifies variance ("volatility drag"), and is a bigger driver of drawdown
than any per-trade edge issue.

## Statistic to optimize

Peak simultaneous capital-at-risk (%), and its direct consequence, max
drawdown (42.56% in the recorded run).

## Concrete changes

- [ ] Add an explicit portfolio-level exposure cap in the day-loop — e.g. skip
  opening new positions once total open notional exceeds some fraction of
  equity, independent of the *daily* `gross_cap_pct=100` (which only
  constrains that day's *new* batch, not the running total across all
  still-open positions from prior days).
- [ ] Consider capping total *concurrent* position count directly (not just
  new-per-day), or shrinking `max_pos_pct` as a function of
  currently-committed capital rather than a flat 15%.

## Verification

`tests/unit/test_kairos_papertrade_loss_repro.py::TestConcurrentExposureAndTurnover`
already pins the current €74.22 cash floor and €9,665 cumulative notional
(48x turnover), so a fix here is directly measurable against that baseline.

## Files

- `strategy/kairos_papertrade.py` (day-loop, lines ~504-553)
