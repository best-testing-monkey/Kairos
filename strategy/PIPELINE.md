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

Three additional stage modes sit outside this 1-5 chain: `auto` (chains stages
1-4 automatically per interval, see "Stage auto" below), `rebuild_disabled`
(DB-maintenance only - recomputes the `disabled_strategies` table from
existing `oracle_results`, see "Auto-disabled strategies" below), and
`finetune_next` (the automated counterpart to manual stage 5: selects the
best not-yet-finetuned profile, trains and backtests a group-specific
checkpoint, and accepts/rejects it against the existing base result - see
"Stage: `finetune_next`" below).

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

**`disabled_strategies`** - one row per currently-disabled strategy per
`(interval, assets)` profile: `interval`, `assets` (sorted CSV, normalized),
`strategy_name`, `avg_pnl_per_trade`, `sharpe`, `signal_count`,
`source_run_id`, `updated_at`, with `PRIMARY KEY (interval, assets,
strategy_name)`. Fully replaced (deleted + re-inserted) per profile on every
`oracle` run - see "Auto-disabled strategies" below.

**`finetuned_models`** - the registry for automated finetuning
(`--stage finetune_next`), one row per `(assets, interval)` profile ever
claimed: `id` (PK), `assets` (sorted CSV, canonical key), `assets_raw` (as
used in `oracle_results`/`model_results`), `interval`, `backtest_period`,
`train_start`/`train_end`, `test_start`/`test_end`, `model_path`, `status`
(`training`/`accepted`/`rejected`/`failed`), `base_run_id`/
`finetuned_run_id`, `base_viable_count`/`ft_viable_count`,
`base_mean_sharpe`/`ft_mean_sharpe`, `created_at`, with
`UNIQUE(assets, interval)`. See "Stage: `finetune_next`" below for the full
lifecycle.

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
- `--disable_min_signals` (int, default `5`) - minimum oracle `signal_count`
  for the `disabled_strategies` auto-disable criterion (also valid with
  `--stage auto`/`--stage rebuild_disabled`) - see "Auto-disabled
  strategies" below.

This stage always runs `kairos_strategies.py` with `--no-prediction`, so it
never touches the GPU or downloads a model - it is safe to run anywhere. It
also always passes `--no_disabled_filter`, so every strategy is scored on
every oracle run regardless of its current disabled status - see
"Auto-disabled strategies" below for why.

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

Note the distinction from `finetune_next` below: manual `--stage finetuned`
just backtests a checkpoint you already have, on assets/interval/period you
choose yourself - it never trains anything and never touches the
`finetuned_models` registry. `finetune_next` is the automated path that
picks the candidate, trains the checkpoint, backtests it via this same
`run_stage_model(stage="finetuned")` code path, and records the outcome.

### Stage: `finetune_next` (automated finetuning and comparison)

```
uv run ./strategy/kairos_pipeline.py --stage finetune_next \
    [--pred_samples N] [--ft_epochs N] [--ft_batch_size N] [--dry_run] \
    [--assets SYM [SYM ...] --interval INTERVAL --backtest_period PERIOD]
```

The automated counterpart to manual stage 5: closes the loop from oracle
discovery to a trained, evaluated, accepted-or-rejected group-specific
Kronos model, one candidate per invocation - meant to be run repeatedly
during idle GPU time. Full narrative walkthrough (when to run, how to read
the verdict, runtime expectations, the yfinance 1h/729-day horizon caveat):
[`docs/playbooks/model-finetuning.md`](../docs/playbooks/model-finetuning.md).

**Selection criteria** (`select_finetune_candidate()`): among
`(assets, interval, backtest_period)` profiles that have both an
`oracle_results` row set and an existing `stage='base'` `model_results` run,
and that are **not** already present in `finetuned_models` under any status
(training/accepted/rejected/failed - matched on the sorted-assets key),
score each by the count of its latest oracle strategies with
`sharpe > 0 AND signal_count >= 3` ("viable-bar" count), tie-broken by the
mean Sharpe of those same strategies. The single top-ranked profile is
selected; if none qualify, `finetune_next` prints a message and exits
cleanly (nothing to do).

**Model directory naming** (`finetune_model_dir()`):
`models/finetuned/{interval}__{SORTED_ASSETS_JOINED_BY_UNDERSCORE}/`, e.g.
`models/finetuned/1h__BTC-USD_ETH-USD_SOL-USD/` - assets are sorted and
joined with `_` (commas become `_`; `-` and `=` are kept as-is, e.g.
`BTC-USD`, `EURUSD=X`). The trainer writes `best_model/` and `final_model/`
inside this directory; the registry's `model_path` records
`<dir>/best_model`. A `metadata.json` mirroring the registry row is also
written here.

**Registry schema**: see the `finetuned_models` table under "Storage" above.
Canonical key is `UNIQUE(assets, interval)` on the *sorted* assets CSV;
`assets_raw` preserves the original ordering used in `oracle_results`/
`model_results` and in the subprocess invocation.

**Train/test periods** (`compute_finetune_periods()`): `test_end = now`,
`test_start = now - backtest_period` (the same window length as the base
run being compared against, re-anchored to "now" - the pipeline doesn't
track a base run's exact calendar dates, only its period label). `train_end
= test_start` (no leakage). `train_start = train_end` minus the data-source
horizon for the interval - 5 years for daily-ish intervals, capped at 729
days for `1h`/`60m` (mirrors `kairos_strategies.fetch_data_raw`'s
`yf_max_days`; the trainer's fetch just returns whatever history actually
exists within that window). All four timestamps are recorded on the
registry row.

**Orchestration** (`run_stage_finetune_next()`):

1. Select the candidate (or use an explicit `--assets`/`--interval` for a
   manual re-queue - see below).
2. Insert a `finetuned_models` row with `status='training'`, claiming the
   `UNIQUE(assets, interval)` slot immediately.
3. Train via a `uv run finetune` subprocess: `--model NeoQuasar/Kronos-base
   --symbol <raw assets...> --interval I --start TRAIN_START --end
   TRAIN_END --device cuda --epochs <--ft_epochs> --batch-size
   <--ft_batch_size> --output-model <model_dir>` - tokenizer frozen, one
   multi-symbol training run covering every asset in the group. A non-zero
   exit marks the row `status='failed'` and stops (the partial model dir is
   kept for post-mortem); the backtest step never runs on failure.
4. Backtest the trained checkpoint via
   `run_stage_model(stage="finetuned", ...)`, using parameters identical to
   the last base run for this profile (same assets/interval/
   backtest_period/`--pred_samples`).
5. **Accept gate** (`compare_finetuned_vs_base()`): compares viable-bar
   counts (`sharpe > 0 AND signal_count >= 3`) and their mean Sharpes
   between the latest `stage='finetuned'` and `stage='base'` runs for this
   profile. Accepted iff `ft_count > base_count`, or
   `ft_count == base_count AND ft_mean_sharpe > base_mean_sharpe`. Otherwise
   rejected.
6. Update the registry row (`status='accepted'`/`'rejected'`, run ids,
   viable counts, mean sharpes), write `<model_dir>/metadata.json`, write an
   empty `<model_dir>/REJECTED` marker file on rejection, and print a
   verdict block.

**`--dry_run`**: prints the selected candidate, computed periods, and the
planned training command, then returns immediately - no registry row, no
directory, no subprocess. Safe to run any time as a sanity check.

**No auto-retry, manual re-queue only**: rejected and failed profiles stay
in `finetuned_models` under that status permanently and are excluded from
future candidate selection. The only way to retry one is to pass the exact
profile explicitly:

```
uv run ./strategy/kairos_pipeline.py --stage finetune_next \
    --assets BTC-USD ETH-USD SOL-USD --interval 1h
```

This bypasses ranking entirely, deletes the existing `finetuned_models` row
for that profile (matched on the sorted-assets key), and reruns the full
train/backtest/compare cycle from scratch.

**Flags**: `--ft_epochs` (default `10`) and `--ft_batch_size` (default `32`)
forward straight to the `uv run finetune` subprocess's `--epochs`/
`--batch-size`; `--pred_samples` (default `100`) is the same flag used
elsewhere in the pipeline, forwarded to the backtest step.

**Not yet wired**: nothing downstream consumes `accepted` models today -
`kairos_signals.py` and the viability report still only ever read
`stage='base'` results. Wiring accepted `finetuned_models` rows into live
signal generation is a declared next step, not part of this stage.

### Stage: `rebuild_disabled` (DB maintenance, no backtest)

```
uv run ./strategy/kairos_pipeline.py --stage rebuild_disabled \
    [--disable_min_signals N]
```

Recomputes the *entire* `disabled_strategies` table, DB-wide, from the
latest `oracle_results` row per `(strategy_name, assets, interval,
backtest_period)` profile - no `--assets`/`--interval`/`--backtest_period`
needed, it walks every profile present in `oracle_results`. Runs no
backtests and touches no GPU. Use it after changing `--disable_min_signals`
(to re-apply the new threshold to already-collected oracle data) or to
backfill/reconcile the table without waiting for the next scheduled oracle
run. Applies the same criterion as `refresh_disabled_strategies()` (see
"Auto-disabled strategies" above) once per profile, and prints a per-profile
`+N disabled, M re-enabled (now K disabled)` line plus a final
`rebuild_disabled done: P profiles processed, T strategies disabled across
all profiles` summary.

- `--disable_min_signals` (int, default `5`) - same flag and meaning as
  `oracle`/`auto`.

```
uv run ./strategy/kairos_pipeline.py --stage rebuild_disabled --disable_min_signals 10
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
| `--disable_min_signals` | `INT` | `5` | Minimum oracle `signal_count` for the `disabled_strategies` auto-disable criterion - see "Auto-disabled strategies" below. |

### How stage auto works

For each interval in `--intervals`:

1. **Universe (stage 1):** Screen the candidate universe unless `--skip_universe` and a prior universe run already exists for this interval.
2. **Correlation (stage 2):** Compute pairwise correlations and greedily cluster symbols into suggested trading groups. Respects `--asset_class` if given.
3. **Per group:** For each group discovered by stage 2:
   - Run stage 3 (oracle) with `--no-prediction` to get a ceiling baseline.
   - Run stage 4 (base) with the real Kronos-base model to get actual performance.
   - Each stage is skipped if results already exist for that `(assets, interval, backtest_period)` tuple, unless `--force` is passed.
4. **Viability report:** After all intervals and groups complete, join the latest oracle and base results and build a consolidated report (see below).

### Per-run prediction cache (overlapping groups)

With overlapping correlation groups, the same symbol can now show up in
several groups within one `--stage auto` run. Each group's stage-4/5
backtest runs as its own subprocess (`run_backtest_subprocess`), so without
caching, identical per-bar Kronos predictions would be recomputed once per
group that contains the symbol.

`run_stage_auto()` creates a temporary per-run cache directory and sets
`KAIROS_PRED_CACHE_DIR` in the environment passed to every group subprocess
it spawns; `strategy/kairos_predcache.py` implements a disk-backed cache
(one `.npz` file per key, so it survives across subprocess boundaries) with
an in-memory LRU on top bounded by a fraction of `/proc/meminfo`'s
`MemAvailable`. The cache key is `(symbol, interval, bar_timestamp,
lookback_len, pred_samples, model_id, content_hash)`, where `content_hash`
is a cheap hash of the lookback window's close prices - so a different or
stale input window for the "same" symbol/bar never collides with an
existing cache entry. The cache directory is deleted (`shutil.rmtree`) when
the auto run finishes, success or failure. Single-stage invocations (e.g.
`--stage base` on its own) never set `KAIROS_PRED_CACHE_DIR`, so behavior is
unchanged when caching isn't in play. The oracle stage (`--no-prediction`)
never calls the prediction path at all, so it is unaffected either way.

### Resumability and `--force`

Before running stage 3 (oracle) or stage 4 (base), the pipeline checks the `oracle_results` or `model_results` table for an existing row with matching `(assets, interval, backtest_period)`. If found and `--force` is off, that stage is logged as skipped and the next stage or group proceeds. This allows long pipelines to resume after a crash without re-running groups that succeeded.

Passing `--force` clears this check and re-runs all stages unconditionally.

A single group's oracle or base failure (any exception type) is caught,
logged in the failure summary, and does not abort the run — remaining
groups are still processed and the viability report is still built at the
end.

### Recovering from a crashed run

If `--stage auto` was killed or crashed partway through (e.g. the CSV shows
oracle stats but no base stats for some groups), no manual cleanup is
needed:

1. **Just rerun the same command** (no `--force`): already-completed
   groups' oracle/base results are detected via the resumability check above
   and skipped; only groups that never finished are (re)processed.
2. **To regenerate the CSV immediately from whatever is already in the DB**,
   without re-running any backtests: add `--report_only`. This is safe to
   run at any time, including mid-crash-recovery, to check current progress.

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
- **Disabled strategies:** Since `oracle` evaluates the full strategy suite regardless of disabled status (see "Auto-disabled strategies" below), a disabled strategy still gets an `oracle_results` row and thus a row in this report - just a non-viable one, since `base`/`finetuned` skip disabled strategies and leave the `base_*` columns `NaN`. The printed summary line at the end of a `base`/`finetuned` run (e.g., "built 42, disabled 5, evaluating 37") shows how many strategies *that* stage excluded.

### Auto-disabled strategies

The `oracle` stage auto-maintains the `disabled_strategies` table (see
"Storage" above), which replaces the old hand-edited `_DISABLED_BY_PROFILE`
dict entirely:

- **Criteria:** a strategy is disabled for its exact `(interval,
  sorted-assets)` profile when its oracle `avg_pnl_per_trade < 0` **and**
  `signal_count >= --disable_min_signals` (default `5`). This is a more
  direct "loses money" measure than the Sharpe-based diagnostic used
  elsewhere in this doc - a strategy can have negative Sharpe with positive
  average PnL (or vice versa on thin samples), so the two don't always
  agree.
- **Full refresh, not a merge:** every `oracle` run deletes and re-derives
  the profile's `disabled_strategies` rows from that run's results
  (`refresh_disabled_strategies()`), so a strategy that turns profitable is
  automatically re-enabled the next time oracle runs for that profile - no
  hand-editing in either direction.
- **Full-suite evaluation enables re-enabling:** to make re-enabling
  possible, `oracle`'s subprocess call always passes `--no_disabled_filter`,
  so every strategy is scored every run regardless of current disabled
  status. `base`/`finetuned` still skip disabled strategies (no wasted GPU
  on strategies already known non-viable), which is why a disabled strategy
  shows up in the viability report with oracle metrics populated but
  `base_*` columns `NaN` rather than being absent from the report entirely.
- **Diff + CSV per run:** `run_stage_oracle()` prints `[disabled] +N newly
  disabled: [...]; M re-enabled: [...]` after each run, and dumps a CSV
  mirror of the profile's current disabled rows to
  `results/oracle_disabled_strategies_<YYYYmmdd_HHMMSS>.csv`.
- **Tuning the threshold:** `--disable_min_signals` is valid with `--stage
  oracle`, `--stage auto`, and `--stage rebuild_disabled`.
- **Backfill/reconcile:** `--stage rebuild_disabled` recomputes the whole
  table from existing `oracle_results` without re-running any backtests -
  see below.
- **Resolution at read time:** `resolve_disabled_strategies(interval,
  assets)` in `strategy/kairos_strategies.py` is DB-backed: an exact profile
  match returns the (possibly empty) set of disabled names for that profile,
  and an empty result is a meaningful "tested and clean" - it does not fall
  through. Only profiles that have never been oracle-tested fall back to the
  hand-curated `_DISABLED_BY_CLASS` per-`(interval, asset_class)` table,
  which remains the only hand-maintained artifact.

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

Manual `--stage finetuned` remains **manual only** — it is not part of the
`--stage auto` chain, since it backtests a checkpoint you already have and
supply via `--model_path`, on assets/interval/period you choose by hand.

This is now distinct from the **automated** `--stage finetune_next` path
(see "Stage: `finetune_next`" above), which *does* run on its own schedule
(one candidate per invocation, meant to be looped/cron'd during idle GPU
time): it selects the candidate itself, trains the checkpoint itself, then
backtests it via the same underlying `run_stage_model(stage="finetuned")`
call manual stage 5 uses, and records accept/reject in the `finetuned_models`
registry. Neither path is part of the `--stage auto` chain — `finetune_next`
has its own invocation path precisely because it needs the `finetuned_models`
registry state (which candidates are already claimed) that `auto` has no
concept of.

The viability report schema is designed to accept a future `finetuned_sharpe`, `finetuned_signals`, etc. column without schema changes. When a finetuned discovery workflow is established, `persist_viability_report()` can be extended to outer-join on `stage='finetuned'` rows alongside oracle and base, adding those columns to the report while keeping the same database table. This is still open — `finetune_next` currently records its own accept/reject verdict in `finetuned_models` rather than folding into the viability report.

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
  for grouping. This can now be a **per-asset-class dict** instead of a
  single float: the module-level default is
  `MIN_ABS_CORR = {"crypto": 0.75, "default": 0.6}`, i.e. crypto pairs need a
  stronger correlation before they're clustered. A pair's effective
  threshold is the *stricter* (max) of its two symbols' class thresholds, so
  a "cross" pair (spanning two asset classes) uses
  `max(threshold(class_a), threshold(class_b))`. Override via
  `--min_abs_corr 0.7` (uniform float, old behavior) or
  `--min_abs_corr crypto=0.8 equity=0.65 default=0.6` (per-class) on
  `--stage correlation` or `--stage auto`; the effective thresholds are
  printed in the stage-2 summary.
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
- Strategies with **negative `avg_pnl_per_trade`** and enough signals
  (`signal_count >= --disable_min_signals`, default `5`) in an `oracle` run
  are automatically written to the `disabled_strategies` table for that
  exact `(interval, assets)` profile - see "Auto-disabled strategies" above.
  `run_stage_oracle()` still prints the negative-Sharpe strategies at the
  end of each run for quick visual review (Sharpe and avg-PnL-based
  disabling don't always agree on thin samples), plus a `[disabled]` diff
  line showing exactly what changed in the table.
- **Caveat - small `signal_count`**: a strategy with `n < 3` signals in a
  given backtest window has a statistically unreliable Sharpe (one or two
  lucky/unlucky trades can swing it arbitrarily). Do not disable a strategy
  based on a single low-`signal_count` oracle run; either extend
  `--backtest_period` to accumulate more signals or treat the result as
  inconclusive until corroborated by another window/asset set.

## 6. Typical workflow

A full discovery cycle, from a fresh universe scan through auto-disabling
underperforming strategies:

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

# 6. Review the printed `[disabled]` diff from step 3 (or query
#    disabled_strategies / the oracle_disabled_strategies_<timestamp>.csv
#    dump, below) - no hand-editing needed, the table is maintained
#    automatically by run_stage_oracle(). After changing
#    --disable_min_signals, or to backfill/reconcile without re-running
#    backtests, recompute the whole table instead:
uv run ./strategy/kairos_pipeline.py --stage rebuild_disabled --disable_min_signals 5
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

# Currently disabled strategies for a given profile
sqlite3 data/pipeline_results.db \
  "SELECT strategy_name, avg_pnl_per_trade, sharpe, signal_count, updated_at
   FROM disabled_strategies
   WHERE interval='1d' AND assets='BTC-USD,ETH-USD,SOL-USD';"

# Compare oracle vs base vs finetuned Sharpe for one strategy across all runs
sqlite3 data/pipeline_results.db \
  "SELECT 'oracle', run_id, sharpe, signal_count FROM oracle_results WHERE strategy_name='cross_asset_rank'
   UNION ALL
   SELECT stage, run_id, sharpe, signal_count FROM model_results WHERE strategy_name='cross_asset_rank';"
```

## 7. Current signals report

`strategy/kairos_signals.py` turns the latest `viability_report` run into an
actionable, human-readable snapshot: for every viable `(strategy, assets,
interval)` row, it runs one latest-bar prediction per `(assets, interval)`
group and reports what each strategy would do *right now*.

```bash
uv run ./strategy/kairos_signals.py \
    [--db data/pipeline_results.db] [--out results/] \
    [--intervals 1d ...] [--pred_samples 100] [--all]
```

- Reads `viability_report` for `run_id = MAX(run_id)`; filters to `viable=1`
  unless `--all` is passed. `--intervals` filters to a subset of intervals.
- Groups viable rows by `(assets, interval)` so the GPU/model does one batched
  prediction (`predict_all_batch`) per group, not one call per strategy.
- Per group: fetches the latest bars for each symbol, predicts once, builds
  the same per-symbol context `_run_day` uses (`returns_window`,
  `realized_vol`, `capital`, etc.), applies the orchestrator's meta-filters,
  and calls `generate_signal()` for every viable strategy in that group.
- Per-group failures (fetch/prediction errors) and per-strategy issues
  (unknown strategy name not in the registry, or a signal blocked by
  meta-filters) are isolated and reported in `## Failures` / `## Skipped`
  footers rather than aborting the whole run.
- Writes `results/kairos_signals_<YYYYmmddHHMM>.md` with:
  - `## Stats` - a table of every strategy that produced >=1 signal (entry,
    stop, target, confidence, expected value, plus the oracle/base viability
    stats carried over from the DB row).
  - `## Signals` - plain-English bullets, e.g. *"Strategy dfa_persistence
    advised **Long** position on BTC-USD for 12% liquidity with SL at
    58,900.00 (-3.1%) and TP at 63,400.00 (+4.2%). Exit by TP/SL."*
- The report path is printed to stdout when the run finishes.
