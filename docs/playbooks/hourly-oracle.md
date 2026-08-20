# Hourly oracle stage

Test every strategy against **perfect (oracle) predictions** on real 1h price
data — the third stage in the `--interval 1h` pipeline, after universe screening
and correlation grouping. This is "if the model predicted the future perfectly,
would this strategy's entry/exit logic actually be profitable?" — a ceiling check
before spending GPU time on real (imperfect) model predictions.

## Prerequisites

- `--stage universe --interval 1h` and `--stage correlation --interval 1h` must
  have completed and left at least one row in `suggested_groups` (see
  [hourly-universe-screen.md](hourly-universe-screen.md) and
  [hourly-correlation.md](hourly-correlation.md)).
- The `1h` filter-threshold presets must be calibrated (`OrchestratorConfig
  .for_interval("1h", ...)`, `_FILTER_PRESETS_BY_INTERVAL["1h"]` in
  `strategy/kairos_orchestrator.py`) — done as of 2026-08-20 (Epic 12, E12-S01/S02):
  entropy_threshold=3.0, kurtosis_max=10.0, min_volume_percentile=10.0, verified
  via a live sweep to match 1d's values.
- GPU not strictly required for oracle mode itself (it runs with
  `--no-prediction`, evaluating strategies against actual next-bar OHLCV rather
  than model output) — but the subprocess still goes through the usual model
  loading/import path, so a working `uv` environment is needed. Real wall-clock
  time: a single-symbol, 1-month backtest period took a few minutes.

## Steps

```bash
# Find a group_id from the correlation stage first:
sqlite3 data/pipeline_results.db \
  "SELECT group_id, symbols FROM suggested_groups WHERE run_id = (SELECT MAX(run_id) FROM runs WHERE stage='correlation' AND interval='1h')"

uv run ./strategy/kairos_pipeline.py --stage oracle --interval 1h --group_id <id> --backtest_period 1m
```

`--backtest_period` defaults to `6m` if omitted; for a first/quick verification
run, a shorter period (`1m`) is enough to confirm the mechanism works and is
much faster.

## What a successful run looks like (observed 2026-08-20, group_id=5, `ZW=F`, 1m period)

```
KAIROS BACKTEST RESULTS
  Mode: NO-PREDICTION  (oracle - actual next-bar OHLCV)
====================================================================
  Total Return:     -5.50%
  Sharpe Ratio:     -6.44
  ...
  Strategies: built 127, disabled 0, evaluating 127 (36 fired at least one signal)
====================================================================
  ALL STRATEGIES BY SHARPE  (shadow: each signal vs actual next-bar):
     1. path_high_low_sequence               Sharpe: 100.00  (4 signals)
     ...
Stage 3 (oracle) done: built 127, disabled 0, evaluating 127 strategies (36 fired
at least one signal). run_id=737. CSV: results/oracle_oracle_results_....csv
Strategies with negative Sharpe (25): ...
[disabled] +25 newly disabled: [...]; 0 re-enabled: []
```

- `oracle_results` (`SELECT * FROM oracle_results WHERE interval='1h' ORDER BY
  run_id DESC`) gets one row per strategy: `avg_pnl_per_trade`, `sharpe`,
  `signal_count`. Sharpe values are bounded by `_safe_sharpe`'s clamp
  (`kairos_orchestrator.py`) — the observed run's max was exactly `100.00`
  (the clamp ceiling, for a strategy with only 4 signals), which is expected
  clamp behavior, not a bug. A `sharpe` far outside the clamp range, or `NaN`,
  would indicate a real problem.
- The `refresh_disabled_strategies` mechanism is live for `1h`: this run alone
  disabled 25 strategies with negative shadow Sharpe for the `(1h, ZW=F)`
  profile, written into `disabled_strategies` (confirmed:
  `SELECT COUNT(*) FROM disabled_strategies WHERE interval='1h'` → 1216 rows
  across all `1h` profiles tested to date). An empty result for a brand-new
  profile is equally valid — it just means nothing's been oracle-tested for
  that specific `(interval, assets)` pair yet.

## Caveats

- **Single-symbol groups are common right now**: as of 2026-08-20, `1h`
  correlation (see [hourly-correlation.md](hourly-correlation.md)) produces
  mostly singleton groups (no pairs cleared the correlation threshold) —
  expect to run oracle one symbol at a time until the universe survivor set
  broadens (crypto is still thin due to BUG-04's residual `$vol=0.0` issue,
  see `docs/todo.md`).
- **Harmless LAPACK warning observed**: the run above printed many `** On
  entry to DLASCL parameter number 4 had an illegal value` lines to stderr.
  This is a numpy/scipy linear-algebra warning, not a crash — the run
  completed and produced correct-looking results. Likely triggered by a
  degenerate (1x1 or near-singular) correlation/covariance matrix computation
  somewhere in a cross-asset-aware strategy when only one symbol is in the
  group. Not investigated further as part of this story; flag it if it starts
  appearing alongside actually wrong numbers, but on its own it isn't one.
- **Next stage**: oracle results (`avg_pnl_per_trade`, `sharpe`, `signal_count`
  per strategy) feed `--stage base --interval 1h` (Epic 13) and, if a strategy
  looks viable there too, `--stage finetuned --interval 1h` (Epic 14) next —
  oracle alone doesn't gate anything by itself, it's informational plus the
  disabled-strategies side effect described above.

See also: [hourly-correlation.md](hourly-correlation.md) (prerequisite stage)
and [model-finetuning.md](model-finetuning.md) (the base/finetuned stages that
follow, and the finetune_next automation).
