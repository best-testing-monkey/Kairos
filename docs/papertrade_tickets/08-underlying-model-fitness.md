# Factor 8: Underlying model/signal fitness

Source: `docs/papertrade_loss_analysis.md` §4, Factor 8

## Problem

The base Kronos model's raw predictive quality per asset, and the gap
between it and the oracle (perfect-foresight) ceiling, is the fundamental
limit on what any downstream tuning (Factors 1-7) can achieve.
`oracle_sharpe`/`oracle_win_rate` are already computed and stored per
strategy/asset in `viability_report` — they're the right yardstick for
where finetuning effort would pay off most.

## Statistic to optimize

The oracle-vs-realized Sharpe gap, per asset, especially for the tickers
that actually dominate the recorded run's losses.

## Concrete changes

- [ ] Prioritize finetuning specifically for `LDO-USD`, `AAVE-USD`,
  `FIL-USD`, `ATOM-USD`, `XTZ-USD`, `AXS-USD` (the top losing tickers here,
  none of which — except indirectly via `BTC-USD` — currently have an
  accepted finetuned model), instead of continuing to only rotate through
  the hardcoded `BTC-USD/ETH-USD/SOL-USD` list.
- [ ] Feed `select_finetune_candidate`'s ranking (already in
  `kairos_pipeline.py`) with a wider `min_signals` sample if these tickers
  currently don't have enough oracle-viable strategies to be considered —
  otherwise they may simply never surface as finetune candidates even though
  they're the biggest realized drag.

## Files

- `strategy/kairos_pipeline.py` (`select_finetune_candidate`, `viability_report`)

## Note

Depends on `06-model-selection-finetuned-overlay.md`'s note that the
hardcoded-rotation-list mechanism has already been removed in favor of
`select_finetune_candidate`'s own ranking.
