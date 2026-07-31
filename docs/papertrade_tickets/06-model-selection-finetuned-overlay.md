# Factor 6: Model selection — base vs. finetuned overlay

Source: `docs/papertrade_loss_analysis.md` §4, Factor 6

## Problem

The recorded 2026-07-23 run used `--base-only` (the CLI default at the
time), so the entire 6-month backtest used only the base Kronos model
(confirmed by `meta.base_only: true` in the recorded JSON). **The default
has since been flipped**: `kairos_papertrade.py` now defaults to the
finetuned-overlay when an accepted model exists, requiring `--base-only` to
explicitly opt back to the base model.

`data/pipeline_results.db` already had 3 `accepted` finetuned models
(covering `ADA-USD/ETH-USD/LINK-USD/SOL-USD`,
`AVAX-USD/LINK-USD/SOL-USD/SUI20947-USD`, and
`ADA-USD/DOT-USD/SUI20947-USD/TIA-USD`) that passed the
realized-backtest-Sharpe accept gate in `kairos_pipeline.py`'s
`compare_finetuned_vs_base` — meaning these models have already been shown
to beat the base model on realized (not oracle) Sharpe for these asset
groups. `SOL-USD` is among the recorded run's traded assets.

## Statistic to optimize

Per-asset-group realized Sharpe/signal-count, base vs. finetuned — the exact
stat the accept gate already tracks in `model_results`.

## Concrete changes

- [x] **In progress:** Re-run the same 6-month window with the *new* default
  (finetuned overlay enabled) to see the impact of using the accepted models
  that have already been vetted as outperforming base on realized backtest
  Sharpe — matches what Kairos's live daily-signals pipeline
  (`kairos_daily_signals.py`) actually recommends. Pin `--effective_per
  "20260723 1458"` so the run replays the identical historical window as the
  original for a true before/after comparison.
  *(A run with this exact command was kicked off in the background on
  2026-07-27; see the doc's §0 for why a naive rerun on a shifted window
  isn't a valid comparison.)*
- [ ] Broaden finetuning candidate coverage beyond a hardcoded rotation list
  (see Factor 8) so several of the run's biggest losing tickers get
  considered. *(Note: the old `kairos_idle_finetune.py` wrapper that rotated
  through a hardcoded `["BTC-USD","ETH-USD","SOL-USD"]` default has since
  been removed; `--stage finetune_next`'s own `select_finetune_candidate`
  ranking, with no hardcoded list, is the mechanism now.)*

## Files

- `strategy/kairos_papertrade.py` (`--base-only`/overlay default)
- `strategy/kairos_pipeline.py` (`compare_finetuned_vs_base`, `select_finetune_candidate`)
- `data/pipeline_results.db` (`model_results`, accepted finetuned models)
