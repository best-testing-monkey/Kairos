# Kairos Asset-Discovery Pipeline

`strategy/kairos_pipeline.py` is a staged CLI tool for discovering which
symbols (out of a hand-curated candidate universe) are liquid and volatile
enough to backtest, which of those symbols move together (candidates for the
`cross_asset_*` strategies in `kairos_meta.py`), and how the strategy suite
actually performs on the survivors - first against a perfect ("oracle")
baseline, then against the real Kronos model (base and finetuned checkpoints).

Run with `uv run ./strategy/kairos_pipeline.py --stage <stage> [options]`.

## 1. Overview

The pipeline runs in five stages, each one feeding the next:

1. **`universe`** - screens `CANDIDATE_UNIVERSE` (crypto / equity /
   fx_commodity symbols hard-coded in `kairos_pipeline.py`) for liquidity,
   volatility, and history length. Writes one row per symbol to
   `universe_screen`, `passed=1` for survivors.
2. **`correlation`** - takes the most recent `universe` run's survivors,
   computes pairwise same-asset-class correlations, and greedily clusters
   highly-correlated symbols into `suggested_groups`. These groups are what
   the `cross_asset_*` strategies (`cross_asset_rank`, `cross_asset_spread`,
   `cross_asset_momentum`, etc. in `kairos_meta.py`) need: a basket of
   symbols that actually co-move.
3. **`oracle`** - takes an explicit `--assets` list or a `--group_id` from
   stage 2, and runs `strategy/kairos_strategies.py` as a subprocess with
   `--no-prediction` (oracle baseline: actual next-bar OHLCV instead of a
   model prediction). No GPU or model download needed. Results land in
   `oracle_results` - this is the "best case" ceiling for each strategy.
4. **`base`** - identical subprocess call, but without `--no-prediction`,
   so it uses the real Kronos model (default: `NeoQuasar/Kronos-base` from
   HF, chosen by `kairos_strategies.py` itself when `--model` is omitted).
   Requires a GPU (or a very patient CPU). Results land in `model_results`
   with `stage='base'`.
5. **`finetuned`** - same as `base`, but forwards `--model_path` from the
   pipeline's CLI as `--model <path>` to `kairos_strategies.py`, pointing at
   a local finetuned Kronos checkpoint. Results land in `model_results` with
   `stage='finetuned'` and `model_path` populated.

Each stage reads its input from the *latest* row of the previous stage's
table in the SQLite DB (e.g. `correlation` reads `MAX(run_id)` from
`universe_screen`), so stages are meant to be run in order at least once,
after which stages 3-5 can be re-run repeatedly with different
`--assets`/`--group_id`/`--interval` combinations without re-running 1-2.

## 2. Storage

### SQLite: `data/pipeline_results.db`

Created (schema applied via `CREATE TABLE IF NOT EXISTS`) on first
connection by `get_connection()`. Tables (columns as declared in `SCHEMA` in
`kairos_pipeline.py`):

**`runs`** - one row per invocation of any stage:
`run_id` (PK, autoincrement), `stage`, `timestamp` (ISO, seconds), `interval`,
`params_json` (JSON dump of the stage's kwargs).

**`universe_screen`** - one row per candidate symbol per `universe` run:
`run_id`, `symbol`, `asset_class`, `bars`, `dollar_volume`, `ann_vol`,
`atr_pct`, `interval_probe_ok` (0/1), `liquidity_note`, `passed` (0/1),
`fail_reason`.

**`correlation_pairs`** - one row per symbol pair per `correlation` run:
`run_id`, `symbol_a`, `symbol_b`, `asset_class`, `full_corr`,
`rolling_corr_median`, `overlap_bars`.

**`suggested_groups`** - one row per greedily-clustered group per
`correlation` run: `run_id`, `group_id`, `asset_class`, `symbols`
(comma-joined string), `mean_intra_corr`.

**`oracle_results`** - one row per strategy per `oracle` run: `run_id`,
`stage` (always `"oracle"`), `strategy_name`, `sharpe`, `signal_count`,
`win_rate`, `avg_pnl_per_trade`, `assets` (comma-joined), `interval`,
`backtest_period`.

**`model_results`** - same shape as `oracle_results` plus `model_path`:
`run_id`, `stage` (`"base"` or `"finetuned"`), `strategy_name`, `sharpe`,
`signal_count`, `win_rate`, `avg_pnl_per_trade`, `assets`, `interval`,
`backtest_period`, `model_path`.

### CSV dumps: `results/`

After each stage, `dump_csv(table, rows, stage)` writes the rows just
inserted to `results/<stage>_<table>_<YYYYmmdd_HHMMSS>.csv` (e.g.
`results/universe_universe_screen_20260705_120000.csv`,
`results/correlation_correlation_pairs_...csv` and
`results/correlation_suggested_groups_...csv` from the same run). The table
name is included because `correlation` writes to two tables per run. These
CSVs are point-in-time mirrors; the DB is the source of truth for
cross-stage joins.

### How `run_id` links stages

`correlation` looks up `SELECT ... FROM universe_screen WHERE passed=1 AND
run_id=(SELECT MAX(run_id) FROM universe_screen)` - i.e. it always uses the
newest universe run, filtered by an optional `--asset_class`. `oracle`/
`base`/`finetuned` do not read `universe_screen` or `correlation_pairs`
directly; when given `--group_id N` they resolve it via
`_group_symbols_from_db()`, which looks up `suggested_groups` for
`group_id=N AND run_id=(SELECT MAX(run_id) FROM suggested_groups)`. There is
no foreign-key enforcement; joins are done manually (see the sqlite3
one-liners below).

## 3. Running each stage

All commands assume `cwd` is the repo root and use `uv run`.

### Stage 1: `universe`

```
uv run ./strategy/kairos_pipeline.py --stage universe [--interval INTERVAL]
```

Flags used by this stage: `--interval` (default `"1d"`) - the daily screen
itself always fetches `1d` bars for the liquidity/volatility computation;
`--interval` is only used to additionally *probe* whether that interval has
any data at all (`interval_probe_ok`), which matters for intraday intervals
like `1h`/`15m` that some data sources do not support for all symbols.

Examples:

```
# Default: daily screen, no intraday probe (interval_probe_ok always True)
uv run ./strategy/kairos_pipeline.py --stage universe

# Screen the universe and additionally check which symbols have 1h data
uv run ./strategy/kairos_pipeline.py --stage universe --interval 1h

# Same for 15-minute bars
uv run ./strategy/kairos_pipeline.py --stage universe --interval 15m
```

### Stage 2: `correlation`

```
uv run ./strategy/kairos_pipeline.py --stage correlation [--asset_class {crypto,equity,fx_commodity}]
```

`--asset_class` restricts the pairwise correlation computation (and the
resulting groups) to one class; omitting it correlates within each class
separately (pairs across classes are never compared - see
`if classes[a] != classes[b]: continue`).

Examples:

```
# Correlate and group across all three asset classes
uv run ./strategy/kairos_pipeline.py --stage correlation

# Only crypto
uv run ./strategy/kairos_pipeline.py --stage correlation --asset_class crypto

# Only FX/commodities
uv run ./strategy/kairos_pipeline.py --stage correlation --asset_class fx_commodity
```

Requires a prior `universe` run with at least one `passed=1` row; otherwise
it prints a message and exits without inserting anything.

### Stage 3: `oracle`

```
uv run ./strategy/kairos_pipeline.py --stage oracle \
    (--assets SYM [SYM ...] | --group_id N) \
    [--interval INTERVAL] [--backtest_period PERIOD] [--pred_samples N]
```

Flags (defaults from `kairos_pipeline.py`'s argparse):
- `--assets` (nargs, required unless `--group_id` given) - explicit symbol list.
- `--group_id` (int, default `None`) - pulls the symbol list from
  `suggested_groups` (latest `correlation` run) instead of `--assets`.
- `--interval` (default `"1d"`)
- `--backtest_period` (default `"6m"`)
- `--pred_samples` (int, default `100`)

This stage always runs `kairos_strategies.py` with `--no-prediction`, so it
never touches the GPU or downloads a model - it is safe to run anywhere.

Examples:

```
# Oracle baseline on the three default assets, daily bars, 6-month window
uv run ./strategy/kairos_pipeline.py --stage oracle --assets BTC-USD ETH-USD SOL-USD

# Same, but on a suggested group discovered by stage 2
uv run ./strategy/kairos_pipeline.py --stage oracle --group_id 3

# Oracle at 1h interval over a shorter 1-month window
uv run ./strategy/kairos_pipeline.py --stage oracle --assets BTC-USD ETH-USD SOL-USD \
    --interval 1h --backtest_period 1m
```

### Stage 4: `base` (real Kronos-base model, GPU required)

```
uv run ./strategy/kairos_pipeline.py --stage base \
    (--assets SYM [SYM ...] | --group_id N) \
    [--interval INTERVAL] [--backtest_period PERIOD] [--pred_samples N]
```

Same flags as `oracle` (no `--no-prediction` is passed, so
`kairos_strategies.py` loads and runs the actual Kronos model - default
`NeoQuasar/Kronos-base` from Hugging Face, since no `--model` is forwarded).
This is CPU-runnable in principle (INT8 dynamic quantization) but expected
to be slow; the pipeline authors note it is "not executed in this
environment" without a GPU.

```
uv run ./strategy/kairos_pipeline.py --stage base --assets BTC-USD ETH-USD SOL-USD
```

### Unattended / overnight runs and GPU recovery

Stages 4/5 (and `--stage auto`) require CUDA by default: `_ensure_model_loaded()`
calls `kairos_gpu.ensure_cuda()`, which invokes `scripts/gpu_recover.py`'s
escalation ladder (free GPU processes -> UVM reload -> full module reload ->
reboot+resume) if torch can't see CUDA. For long overnight discovery runs where
no one is present to approve a reboot, set `KAIROS_GPU_ALLOW_REBOOT=1` so the
ladder is allowed to reach L4 and reboot+resume the pipeline automatically; the
resume unit re-runs the requesting command after the next login (or immediately
if `loginctl enable-linger` is set for the user). Set `KAIROS_ALLOW_CPU=1` to
opt back into the old silent CPU fallback instead of invoking recovery. A
subprocess that exits `75` (EX_TEMPFAIL - GPU was just healed but this
process's cached torch state is stale) is retried exactly once by
`run_backtest_subprocess`.

### Stage 5: `finetuned` (finetuned checkpoint, GPU required)

```
uv run ./strategy/kairos_pipeline.py --stage finetuned \
    (--assets SYM [SYM ...] | --group_id N) \
    --model_path PATH \
    [--interval INTERVAL] [--backtest_period PERIOD] [--pred_samples N]
```

`--model_path` (default `None`) is a pipeline-only flag. `kairos_pipeline.py`
has no separate concept of a "finetuned path" flag in the subprocess it
calls - `kairos_strategies.py` only exposes `--model`, which is normally used
for local finetuned checkpoints (see its help text: "Local path to finetuned
Kronos predictor"). The pipeline forwards `--model_path`'s value as
`--model <path>` to the subprocess only when `--stage finetuned`; for `base`
runs, `model_path` is always `None` so `kairos_strategies.py` falls back to
its own default (`NeoQuasar/Kronos-base`).

```
uv run ./strategy/kairos_pipeline.py --stage finetuned --assets BTC-USD ETH-USD SOL-USD \
    --model_path /path/to/finetuned_kronos_checkpoint
```

## Stage auto: Unified discovery pipeline

For a complete discovery cycle in one command, use `--stage auto` to chain stages 1–4
in order: universe → correlation → oracle → base for each requested bar interval.

```
uv run ./strategy/kairos_pipeline.py --stage auto \
    [--intervals 1d [1h ...]] [--backtest_period 6m] [--asset_class crypto] \
    [--pred_samples 100] [--min_sharpe 0.0] [--min_signals 3] \
    [--force] [--skip_universe] [--report_only]
```

### Flags

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--intervals` | `STR [STR ...]` | `["1d"]` | Bar intervals to test (e.g., `1d`, `1h`, `15m`); repeats the full chain once per interval. |
| `--backtest_period` | `STR` | `"6m"` | Lookback window passed to stages 3–4 (oracle and base). |
| `--asset_class` | `{crypto, equity, fx_commodity}` | `None` (all) | Optional filter: only test assets in this class. |
| `--pred_samples` | `INT` | `100` | Number of prediction samples for stochastic inference. |
| `--min_sharpe` | `FLOAT` | `0.0` | Viability threshold: oracle *and* base Sharpe must exceed this. |
| `--min_signals` | `INT` | `3` | Viability threshold: both sides must have ≥ this many trade signals. |
| `--force` | `FLAG` | off | Force re-run of oracle/base, even if results already exist for an (assets, interval, backtest_period) key. |
| `--skip_universe` | `FLAG` | off | Skip stage 1; reuse the latest existing universe/correlation run per interval. Useful after a crash. |
| `--report_only` | `FLAG` | off | Skip stages 1–4 entirely; rebuild the viability report from existing DB rows matching the other flags. |

### How stage auto works

For each interval in `--intervals`:

1. **Universe (stage 1):** Screen the candidate universe unless `--skip_universe` and a prior universe run already exists for this interval.
2. **Correlation (stage 2):** Compute pairwise correlations and greedily cluster symbols into suggested trading groups. Respects `--asset_class` if given.
3. **Per group:** For each group discovered by stage 2:
   - Run stage 3 (oracle) with `--no-prediction` to get a ceiling baseline.
   - Run stage 4 (base) with the real Kronos-base model to get actual performance.
   - Each stage is skipped if results already exist for that `(assets, interval, backtest_period)` tuple, unless `--force` is passed.
4. **Viability report:** After all intervals and groups complete, join the latest oracle and base results and build a consolidated report (see below).

### Resumability and `--force`

Before running stage 3 (oracle) or stage 4 (base), the pipeline checks the `oracle_results` or `model_results` table for an existing row with matching `(assets, interval, backtest_period)`. If found and `--force` is off, that stage is logged as skipped and the next stage or group proceeds. This allows long pipelines to resume after a crash without re-running groups that succeeded.

Passing `--force` clears this check and re-runs all stages unconditionally.

### Viability report

After all intervals and groups are processed (or immediately with `--report_only`), a **viability report** is generated:

- **Location:** `results/auto_viability_report_<YYYYmmdd_HHMMSS>.csv` and the `viability_report` SQLite table.
- **Columns (in order):** strategy_name, assets, asset_class, interval, backtest_period, oracle_sharpe, oracle_signals, oracle_win_rate, oracle_avg_pnl_per_trade, oracle_run_id, base_sharpe, base_signals, base_win_rate, base_avg_pnl_per_trade, base_run_id, base_model_path, signals_per_week, viable.
- **viability rule:** A strategy is marked `viable=True` only if:
  - `oracle_sharpe > min_sharpe` **AND**
  - `base_sharpe > min_sharpe` **AND**
  - `min(oracle_signals, base_signals) >= min_signals`.
  
  Any row with NaN on either side defaults to `viable=False`.
- **signals_per_week:** Computed as `base_signals / (backtest_period_in_weeks)`, where `6m` ≈ 26.1 weeks, `1m` ≈ 4.35 weeks, etc. Falls back to `oracle_signals` if base is NaN.
- **Sort:** Viable strategies first, then by `base_sharpe` descending.
- **Disabled strategies:** The report covers *enabled* strategies only. Disabled strategies (as per `resolve_disabled_strategies` in `kairos_strategies.py`) never appear in the results tables and thus never appear in the report. The printed summary line (e.g., "built 42, disabled 5, evaluating 37") shows the excluded count.

### Example: crypto discovery over 3 months and 1 day

```
uv run ./strategy/kairos_pipeline.py --stage auto \
    --intervals 1d \
    --backtest_period 3m \
    --asset_class crypto
```

This chains universe → correlation → oracle → base for all crypto assets, backtesting each discovered group over a 3-month window at daily bars. The viability report is written to `results/auto_viability_report_<timestamp>.csv` and the `viability_report` table.

To add an intraday interval:

```
uv run ./strategy/kairos_pipeline.py --stage auto \
    --intervals 1d 1h \
    --backtest_period 3m \
    --asset_class crypto
```

This runs the chain twice: once for daily bars, once for hourly bars. Both intervals appear in the report.

### Stage 5 (finetuned) and future extensions

Stage 5 (finetuned) remains **manual only** — it is not part of the auto chain, as finetuned checkpoints vary per experiment and are not part of the standard discovery flow.

The viability report schema is designed to accept a future `finetuned_sharpe`, `finetuned_signals`, etc. column without schema changes. When a finetuned discovery workflow is established, `persist_viability_report()` can be extended to outer-join on `stage='finetuned'` rows alongside oracle and base, adding those columns to the report while keeping the same database table.

## 4. Screening criteria (from `kairos_pipeline.py` constants)

### Liquidity (`liquidity_threshold()`)

| Asset class | Minimum median daily dollar volume |
|---|---|
| `crypto` | $10,000,000 |
| `equity` | $50,000,000 |
| `fx_commodity` | 0 (see FX exemption below) |

### FX volume exemption

Symbols ending in `=X` (the `FX_SUFFIX`) are treated as `is_fx` and are
**exempt from the dollar-volume filter entirely** - yfinance reports
zero/NaN volume for FX pairs. Instead, `evaluate_liquidity()` records
`liquidity_note = "fx_exempt_from_dollar_volume_filter"` and only applies
the bar-count and ATR% checks. Note this exemption is keyed off the ticker
suffix, not the `fx_commodity` class as a whole - so commodity ETFs/futures
in that class (e.g. `GLD`, `GC=F`) that don't end in `=X` still go through
the normal dollar-volume check with a threshold of `0.0` (i.e. effectively
also unconstrained, since `fx_commodity`'s `liquidity_threshold()` return is
`0.0` for the whole class regardless of suffix).

### Other floors (`evaluate_liquidity()` defaults)

- `min_bars = 200` - symbols with fewer than 200 daily bars in the ~400-day
  lookback window fail with `insufficient_bars(...)`.
- `atr_min = 0.5` - a 14-period ATR below 0.5% of the last close fails with
  `low_atr_pct(...)`. This is the volatility floor; too-quiet symbols are
  excluded regardless of liquidity.

### Correlation and grouping (`compute_pair_correlation()`, `greedy_group_pairs()`)

- `min_overlap = 150` bars - pairs with fewer than 150 overlapping close
  prices are skipped (correlation not computed, pair not inserted).
- `roll_window = 30` - the rolling correlation window used for
  `rolling_corr_median` (informational; grouping uses `full_corr`, the
  correlation of full-history log returns, not the rolling median).
- `min_abs_corr = 0.6` - only pairs with `|full_corr| >= 0.6` are eligible
  for grouping.
- `max_group_size = 4` - groups stop absorbing new symbols once they reach
  4 members.
- Grouping algorithm: sort strong pairs by `|corr|` descending, then
  greedily merge each pair's two symbols into a shared group (creating one
  if neither symbol is grouped yet, extending an existing group if exactly
  one symbol already belongs to one and it has room, or just recording an
  extra intra-group correlation if both symbols are already in the *same*
  group). Pairs whose two symbols are already in two *different* groups are
  not merged - the algorithm never merges existing groups, to keep behavior
  simple and deterministic.

## 5. Interpreting results

- **Shadow Sharpe** (the `sharpe` column in `oracle_results`/`model_results`,
  sourced from `payload["shadow_performance"][strategy]["sharpe"]`) measures
  each strategy's signals "in the shadows" - i.e. every signal a strategy
  would have emitted is scored against the *actual* next-bar outcome,
  independent of whether that strategy was live/ranked/executed on a given
  day. It's the per-strategy diagnostic signal used to decide which
  strategies are worth keeping for a given `(interval, assets)` profile.
- Strategies with **negative Sharpe** in an `oracle` run are the primary
  candidates for the `_DISABLED_BY_PROFILE` table in `kairos_strategies.py`
  (see existing entries keyed by `(interval, "SYM1,SYM2,SYM3")` - e.g.
  `("1d", "BTC-USD,ETH-USD,SOL-USD")`). `run_stage_oracle()` prints the
  negative-Sharpe strategies at the end of each run specifically to make
  this easy to review and copy into the profile dict.
- **Caveat - small `signal_count`**: a strategy with `n < 3` signals in a
  given backtest window has a statistically unreliable Sharpe (one or two
  lucky/unlucky trades can swing it arbitrarily). Do not disable a strategy
  based on a single low-`signal_count` oracle run; either extend
  `--backtest_period` to accumulate more signals or treat the result as
  inconclusive until corroborated by another window/asset set.

## 6. Typical workflow

A full discovery cycle, from a fresh universe scan to updating
`_DISABLED_BY_PROFILE`:

```
# 1. Screen the full candidate universe (daily bars)
uv run ./strategy/kairos_pipeline.py --stage universe

# 2. Correlate survivors and get suggested cross-asset groups
uv run ./strategy/kairos_pipeline.py --stage correlation

# 3. Oracle-backtest a specific asset set or a discovered group
uv run ./strategy/kairos_pipeline.py --stage oracle --assets BTC-USD ETH-USD SOL-USD
# ...or, using a group_id printed by stage 2:
uv run ./strategy/kairos_pipeline.py --stage oracle --group_id 2

# 4. (GPU) Confirm on the real base model
uv run ./strategy/kairos_pipeline.py --stage base --assets BTC-USD ETH-USD SOL-USD

# 5. (GPU) Confirm on a finetuned checkpoint, if available
uv run ./strategy/kairos_pipeline.py --stage finetuned --assets BTC-USD ETH-USD SOL-USD \
    --model_path /path/to/checkpoint

# 6. Review negative-Sharpe strategies from the oracle run's stdout (or query
#    the DB, below), then hand-edit _DISABLED_BY_PROFILE in
#    strategy/kairos_strategies.py with a new/updated
#    (interval, "SYM1,SYM2,...") key.
```

Useful `sqlite3` one-liners against `data/pipeline_results.db`:

```bash
# Most recent run per stage
sqlite3 data/pipeline_results.db \
  "SELECT run_id, stage, timestamp, interval FROM runs ORDER BY run_id DESC LIMIT 10;"

# Symbols that passed the latest universe screen
sqlite3 data/pipeline_results.db \
  "SELECT symbol, asset_class, dollar_volume, atr_pct FROM universe_screen
   WHERE passed=1 AND run_id=(SELECT MAX(run_id) FROM universe_screen)
   ORDER BY asset_class, symbol;"

# Latest suggested cross-asset groups
sqlite3 data/pipeline_results.db \
  "SELECT group_id, asset_class, symbols, mean_intra_corr FROM suggested_groups
   WHERE run_id=(SELECT MAX(run_id) FROM suggested_groups);"

# Negative-Sharpe strategies from the most recent oracle run, sorted worst-first
sqlite3 data/pipeline_results.db \
  "SELECT strategy_name, sharpe, signal_count, avg_pnl_per_trade FROM oracle_results
   WHERE run_id=(SELECT MAX(run_id) FROM oracle_results) AND sharpe < 0
   ORDER BY sharpe ASC;"

# Compare oracle vs base vs finetuned Sharpe for one strategy across all runs
sqlite3 data/pipeline_results.db \
  "SELECT 'oracle', run_id, sharpe, signal_count FROM oracle_results WHERE strategy_name='cross_asset_rank'
   UNION ALL
   SELECT stage, run_id, sharpe, signal_count FROM model_results WHERE strategy_name='cross_asset_rank';"
```
