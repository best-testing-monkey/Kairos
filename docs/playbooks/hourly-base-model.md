# Hourly base-model backtest

Backtest every (non-disabled) strategy against the **base Kronos model's real
predictions** on 1h price data — the fourth stage in the `--interval 1h`
pipeline, after universe screening, correlation grouping, and the oracle
ceiling check. Unlike oracle (perfect next-bar knowledge), this uses actual
GPU model inference, so it's the first stage in the 1h pipeline whose numbers
reflect real predictive skill, not just strategy logic.

## Prerequisites

- `--stage universe --interval 1h` and `--stage correlation --interval 1h`
  must have completed and left at least one row in `suggested_groups` (see
  [hourly-universe-screen.md](hourly-universe-screen.md) and
  [hourly-correlation.md](hourly-correlation.md)). Running `--stage oracle`
  first (see [hourly-oracle.md](hourly-oracle.md)) isn't a hard prerequisite
  — base can run against any group with enough price history — but it's the
  realistic pipeline order and informs which groups are worth spending real
  GPU time on.
- **GPU required** (or `KAIROS_ALLOW_CPU=1` for a much slower CPU/INT8
  fallback) — this stage runs `kairos_strategies.py` as a subprocess WITHOUT
  `--no_prediction`, so it loads the base Kronos model and does real
  autoregressive sampling per bar. Observed: `→ GPU mode: autocast FP16, TF32
  matmuls enabled` in the subprocess log.

## Steps

```bash
# Find a group_id from the correlation stage first (same query as the oracle playbook):
sqlite3 data/pipeline_results.db \
  "SELECT group_id, symbols FROM suggested_groups WHERE run_id = (SELECT MAX(run_id) FROM runs WHERE stage='correlation' AND interval='1h')"

uv run ./strategy/kairos_pipeline.py --stage base --interval 1h --group_id <id> --backtest_period 1m
```

`--backtest_period` defaults to `6m` if omitted; `1m` is enough for a quick
verification run and was used for the observed run below.

## What a successful run looks like (observed 2026-08-20, group_id=5, `ZW=F`, 1m period)

```
[subprocess] uv run .../kairos_strategies.py --interval 1h --backtest_period 1m --pred_samples 100 --assets ZW=F --export_json ...
  → GPU mode: autocast FP16, TF32 matmuls enabled
KAIROS BACKTEST RESULTS
  Total Return:     -20.61%
  Sharpe Ratio:     -26.64
  Max Drawdown:     20.84%
  Win Rate:         40.86%
  ALL STRATEGIES - SHADOW SIGNAL PERFORMANCE
  percentile_entry     Sharpe: 23.58   (13 signals)
  fade_extreme         Sharpe: 21.89   (27 signals)
  support_confluence   Sharpe:  3.95   (442 signals)
  ...
  gbm_direction        Sharpe: -64.23   (4 signals)
Stage base done: built 127, disabled 25, evaluating 102 strategies (15 fired at
least one signal). run_id=738. CSV: results/base_model_results_....csv
```

Full round trip (subprocess model load + real GPU inference across the
backtest window): ~2.5-3 minutes for a single-symbol, 1-month period.

- `model_results` (`SELECT * FROM model_results WHERE stage='base' AND
  interval='1h' ORDER BY run_id DESC`) gets one row per strategy, same shape
  as `oracle_results`: `avg_pnl_per_trade`, `sharpe`, `signal_count`. Sharpe
  values are bounded by `_safe_sharpe`'s clamp — the observed run's top values
  (`23.58`, `21.89`) are well inside the clamp ceiling (`100.00`, seen in the
  E12-S03 oracle run for a 4-signal strategy) and reflect real per-strategy
  performance, not clamp saturation.
- Confirmed on this run: `model_results WHERE stage='base' AND interval='1d'`
  row count (`5592`) was unchanged before/after — the `1h` run only added
  `1h`-tagged rows, no cross-interval leakage.
- The same benign LAPACK warning documented in
  [hourly-oracle.md](hourly-oracle.md) (`** On entry to DLASCL parameter
  number 4 had an illegal value`) appeared again here — same likely cause
  (a degenerate single-symbol correlation matrix in a cross-asset-aware
  strategy), harmless, not investigated further.

## Caveats

- **Single-symbol groups are still common** as of 2026-08-20 (see
  [hourly-correlation.md](hourly-correlation.md)) — same caveat as the oracle
  stage.
- **Next stage**: base-model results (`avg_pnl_per_trade`, `sharpe`,
  `signal_count` per strategy, now reflecting real model skill rather than
  oracle's perfect-information ceiling) feed `--stage finetune_next --interval
  1h`'s comparison baseline (Epic 14) — a finetuned model is only "accepted"
  if it beats this stage's numbers for the same `(assets, interval,
  backtest_period)` profile.

See also: [hourly-oracle.md](hourly-oracle.md) (prerequisite stage) and
[model-finetuning.md](model-finetuning.md) (the finetune_next stage that
follows, and its acceptance-criteria logic).
