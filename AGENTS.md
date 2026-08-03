# Kairos — Agent Onboarding

This file is a concise, accurate guide for AI coding agents working on the Kairos repository. It reflects the actual project layout, tooling, and conventions as of the latest checkout.

---

## 1. Project Overview

**Kairos** is an integration and application layer around two external systems:

- **Kronos** (`model/`) — an open-source foundation model for financial OHLCV time-series forecasting.
- **price_cache** (external dependency) — a gap-aware OHLCV cache with a multi-provider fallback chain.

Kairos sits between them: it pulls historic price data from `price_cache`, adapts it to the input contract Kronos expects, runs forecasts, and then feeds those forecasts into a large strategy/backtesting/pipeline framework.

The repository also contains:

- A **strategy/backtesting engine** (`strategy/`) with 40+ trading strategies, an orchestrator, meta-filters, and a multi-stage asset-discovery pipeline.
- **Fine-tuning pipelines** (`finetune/`, `finetune_csv/`) for training Kronos tokenizers and predictors on custom CSV data, including a Kronos-large distillation path.
- A **Flask web UI** (`webui/`) for interactive forecasting.
- **Example scripts** (`examples/`) for data fetching, prediction, and backtesting.

License: MIT (inherited from upstream Kronos).

---

## 2. Technology Stack

- **Language**: Python 3.11+
- **Package / dependency manager**: `uv` (lockfile `uv.lock`)
- **Build backend**: `hatchling` (declared in `pyproject.toml`)
- **Deep-learning framework**: PyTorch 2.x, with CUDA 12.1 index configured for `torch`
- **Model hub**: Hugging Face `transformers` / `huggingface-hub`
- **Data processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, plotly
- **Technical analysis**: `ta`
- **Exchange calendars**: `exchange-calendars`
- **Spreadsheets**: `openpyxl`, `odfpy`, `gspread` (Google Sheets OAuth)
- **Test runner**: pytest
- **Lint / type check**: flake8, mypy

Key external Python dependencies installed directly from Git:

- `price_cache` (`git+https://github.com/best-testing-monkey/price_cache.git`)
- `phantom-ledger` (`git+https://github.com/best-testing-monkey/phantom_ledger.git`, imported as `phantom`) — sibling paper-trading engine used by `strategy/kairos_papertrade.py`

Also: `sqlitedict` (persistent dict-on-SQLite, used for papertrade report de-dup).

---

## 3. Repository Layout

```
├── kairos/                 # Main installable Python package
│   ├── adapter.py          # price_cache DataFrame → Kronos OHLCV contract
│   ├── calendars.py        # exchange-calendar-aware future timestamp synthesis
│   ├── config.py           # configure() facade over price_cache + calendar state
│   ├── data.py             # get_forecast_window() public entry point
│   ├── errors.py           # KairosError hierarchy
│   ├── windowing.py        # bar-count → date-range windowing with retry
│   └── cli/                # Console entry points
│       ├── forecast.py     # `forecast` command
│       ├── finetune.py     # `finetune` command
│       └── _models.py      # short-name → HuggingFace ID registry
│
├── model/                  # Kronos model, tokenizer, predictor
│   ├── __init__.py
│   ├── kronos.py           # Kronos transformer, tokenizer, predictor
│   └── module.py           # Transformer building blocks
│
├── strategy/               # Trading strategies & backtesting (NOT a Python package)
│   ├── kairos_backtest.py
│   ├── kairos_orchestrator.py
│   ├── kairos_meta.py
│   ├── kairos_execution.py
│   ├── kairos_path.py
│   ├── kairos_horizon.py
│   ├── kairos_pipeline.py  # 5-stage asset-discovery pipeline
│   ├── kairos_signals.py   # Current-signals report generator
│   ├── kairos_papertrade.py # Paper-trade executor (Phantom Ledger, roadmap Phase 4)
│   ├── kairos_predcache.py # Disk-backed prediction cache + in-memory LRU
│   ├── signal_selection.py # --signal-selection DSL grammar + column registry
│   ├── allocation.py       # Signal gating/ranking (default min_n + EV gate, top-K)
│   ├── kairos_gpu.py       # CUDA recovery helpers
│   ├── PIPELINE.md         # Pipeline usage docs
│   └── README.md           # Strategy framework docs
│
├── tests/
│   ├── conftest.py         # Adds strategy/ to sys.path for tests
│   ├── test_kronos_regression.py
│   ├── unit/               # 200+ unit tests, no GPU/network required
│   ├── integration/        # Local SQLite round-trip tests
│   └── data/               # Fixture CSVs for regression tests
│
├── examples/               # Standalone prediction / fetch / backtest scripts
│   └── akshare/            # akshare-based Chinese-market variants
│
├── finetune/               # Upstream-style Kronos finetuning utilities
├── finetune_csv/           # Custom CSV finetuning pipeline + configs
├── webui/                  # Flask web interface
├── scripts/                # gpu_recover.py, smoke.py, automation runners
│   ├── kairos_daily_signals.py     # daily signals runner + Telegram alerts
│   ├── kairos_weekly_discovery.py  # weekly discovery runner + Telegram alerts
│   └── gpu_recover.py              # GPU recovery ladder
├── systemd/                # User systemd service/timer files for automation
├── kairos/                 # Main installable Python package
│   ├── ops.py              # GPU lock, GPU health/utilization, Telegram notifications
│   └── ...
├── data/                   # SQLite DBs, predcache/, phantom_ledger/ (ignored by git)
├── results/                # Pipeline CSV reports (ignored by git)
├── output/                 # Example / report outputs (ignored by git)
├── docs/                   # RFCs, tickets, todo, papertrade_tickets/, playbooks/
└── roadmap/                # Phase documents
```

Important: `strategy/` deliberately has **no `__init__.py`**. Scripts and tests that use it must add `strategy/` to `sys.path` explicitly. `tests/conftest.py` does this for the test suite.

---

## 4. Build, Run, and Development Commands

All routine commands use `uv`.

### Install / sync dependencies

```bash
uv sync
```

### Run the public CLI commands

```bash
# Forecast a symbol
uv run forecast --model kronos-small --symbol AAPL --interval 1d --lookback 64 --pred-len 8

# Fine-tune on a symbol's price history
uv run finetune --model kronos-small --symbol AAPL --output-model ./aapl-model
```

`forecast` and `finetune` are declared as `[project.scripts]` in `pyproject.toml`.

### Run the strategy backtest demo

```bash
uv run ./strategy/kairos_strategies.py
```

This needs a GPU for realistic speed or will fall back to a very slow CPU/INT8 path.

### Run the asset-discovery pipeline

```bash
# Full discovery chain: universe → correlation → oracle → base
uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1d --asset_class crypto

# Individual stages
uv run ./strategy/kairos_pipeline.py --stage universe
uv run ./strategy/kairos_pipeline.py --stage correlation
uv run ./strategy/kairos_pipeline.py --stage oracle --assets BTC-USD ETH-USD SOL-USD
```

See `strategy/PIPELINE.md` for the complete stage reference.

### Generate current signals report

```bash
uv run ./strategy/kairos_signals.py
uv run ./strategy/kairos_signals.py --gsheets   # uploads to Google Sheets
uv run ./strategy/kairos_signals.py --xlsx --ods
uv run ./strategy/kairos_signals.py --signal-selection "'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %' DESC, TOP 3"
```

### Run the paper-trade executor (Phantom Ledger)

```bash
uv run ./strategy/kairos_papertrade.py --months-back 6 --interval 1d
uv run ./strategy/kairos_papertrade.py --no-pred-cache   # bypass the prediction prewarm cache
```

Replays `kairos_signals.py` reports through Phantom Ledger with a one-report lag (report `i`'s candidates execute at report `i+1`'s next-bar open). Sends Telegram lifecycle/slow-iteration alerts (`--no-telegram` to mute); see the papertrade gotchas in §7.

### Start the web UI

```bash
cd webui
python run.py       # or ./start.sh, or python app.py
# then open http://localhost:7070
```

### Fine-tuning (CSV pipeline)

```bash
cd finetune_csv
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml
python generate_distilled_tokens.py --config configs/my_large_run.yaml
python train_large_model.py --config configs/my_large_run.yaml
```

### Scheduled automation runners

```bash
# Daily signals report (after daily bar close)
uv run ./scripts/kairos_daily_signals.py

# Weekly strategy-discovery pass
uv run ./scripts/kairos_weekly_discovery.py

# Weekly discovery, daily interval only (default; add --include-hourly for 1h pass)
uv run ./scripts/kairos_weekly_discovery.py

# Include hourly discovery pass as well
uv run ./scripts/kairos_weekly_discovery.py --include-hourly
```

Both runners:

- Acquire a shared GPU lock so only one Kairos GPU job runs at a time.
- Verify CUDA health (and run the recovery ladder if needed).
- Send Telegram alerts on actionable signals, failures, or completions.

Install the systemd timers from `systemd/`; see `systemd/README.md` for copy/paste commands. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `~/.config/kairos/kairos.env`.

---

## 5. Testing Instructions

### Fast unit tests (no GPU, no network, no model download)

```bash
uv run --with pytest python -m pytest tests/unit/ -q
```

### Run a specific test file

```bash
uv run --with pytest python -m pytest tests/unit/test_kairos_distribution.py -v
```

### Integration tests (local SQLite only)

```bash
uv run --with pytest python -m pytest tests/integration/ -v
```

### Kronos regression tests (downloads a small pinned model from Hugging Face)

```bash
uv run --with pytest python -m pytest tests/test_kronos_regression.py -v
```

### Smoke test (no GPU/network)

```bash
uv run --with pytest python scripts/smoke.py
```

### Key test conventions

- `tests/conftest.py` adds the repo root to `sys.path`.
- `tests/conftest.py` inside `tests/` adds `strategy/` to `sys.path` so strategy modules can be imported by bare name.
- Unit tests should remain independent of GPU, network, and model downloads.
- Use synthetic fixtures; `tests/integration/conftest.py` seeds a temporary SQLite DB with fixture OHLCV data.

---

## 6. Code Style Guidelines

- **Line length**: max 120 characters (`tool.flake8.max-line-length = 120`).
- **Type hints**: use Python 3.11 annotations; prefer `str | None` union syntax.
- **Imports**: group standard library, third-party, and local imports; use absolute imports inside `kairos/`.
- **Docstrings**: modules and public functions have docstrings; many modules are tagged with `KAI-N` ticket identifiers (e.g. `KAI-5` for `data.py`).
- **Error handling**: raise typed exceptions from `kairos.errors` rather than generic `ValueError`/`RuntimeError` for Kairos-specific failures.
- **No `__init__.py` in `strategy/`**: scripts and tests must mutate `sys.path` to import strategy modules by bare name.
- **Formatting**: no explicit formatter is configured; follow the existing flake8 / mypy setup.

### Useful checks

```bash
uv run --with flake8 python -m flake8 kairos/ tests/
uv run --with mypy python -m mypy kairos/
```

---

## 7. Development Conventions and Gotchas

### Strategy module imports

Strategy modules (`kairos_backtest`, `kairos_orchestrator`, `kairos_meta`, etc.) import each other by bare name. Any script or test that uses them must prepend `strategy/` to `sys.path`.

### Signal return types

All `generate_signal()` implementations must return either a `Signal` dataclass (from `kairos_backtest`) or `None`. Returning a plain `dict` breaks the `LiquidityFilterStrategy` wrapper, which accesses `.metadata`.

### Entropy calculation

`KairosDistribution.entropy()` computes **Shannon entropy** in nats (range roughly 0–ln(20) ≈ 3.0). The default `entropy_threshold` is calibrated to 3.0. Do not use `density=True` in `np.histogram` for this filter — differential entropy gives values ~12–14 and would block every asset.

### Kurtosis filter

`kurtosis_max` defaults to 10.0 (excess kurtosis, Fisher definition). Discrete token sampling routinely produces excess kurtosis above 3, so lowering this to 3.0 would silence directional strategies.

### Percentile key format

`_compute_stats()` stores percentile keys as `"pct_10"` (integer, no decimal). Strategy parameters like `stop_pct: float = 10.0` must be cast to `int` before formatting: `f"pct_{int(self.stop_pct)}"`.

### GPU / CUDA behavior

- `model/kronos.py` uses `torch.autocast('cuda', float16)` automatically when on CUDA.
- `torch.compile` is **not supported** on Python 3.13 (`Dynamo is not supported`).
- By default CUDA is required at runtime; if torch cannot see CUDA, `kairos_gpu.ensure_cuda()` runs an escalation ladder via `scripts/gpu_recover.py`.
- Set `KAIROS_ALLOW_CPU=1` to opt back into silent CPU/INT8 fallback.
- Set `KAIROS_GPU_ALLOW_REBOOT=1` to allow the L4 reboot step for unattended runs.
- If a subprocess exits code `75` (GPU healed but current torch process still cannot see it), the pipeline retries it once.

### Prediction samples and lookback

`PRED_SAMPLES = 100` and `DEMO_LOOKBACK = 300` in `strategy/kairos_strategies.py` are hard constraints. Do not reduce them as a performance shortcut.

### Configurable signal selection

`kairos_signals.py`/`kairos_papertrade.py`'s `--signal-selection "<rule>"` flag (grammar in `strategy/signal_selection.py`) lets you replace `strategy/allocation.py`'s hardcoded `min_n`/positive-EV gate and `score`-based ranking/top-K with your own filter+sort rule. When set, the rule **fully replaces** the default gate rather than adding to it — a rule that doesn't check EV can admit a negative-EV signal, so include an EV/quality condition if you want that safety back.

### Paper trading & prediction caches

`strategy/kairos_papertrade.py` replays `kairos_signals.py` reports through Phantom Ledger (roadmap Phase 4). It grew several caches and ops safeguards that are easy to trip over:

- **Persistent prediction cache** (`strategy/kairos_predcache.py`): disk-backed `.npz` store at `data/predcache/` (env `KAIROS_PRED_CACHE_DIR`) plus a byte-bounded in-memory LRU on top. Keys include symbol/interval/bar/lookback/`pred_len`/`pred_samples`/model/content-hash/`checkpoint_fingerprint` — the fingerprint (size+mtime of `model.safetensors`) busts the cache when a finetuned checkpoint is retrained in place. Disk is bounded (default 2 GiB, oldest-mtime eviction; `KAIROS_PRED_CACHE_MAX_BYTES` to tune). `kairos_papertrade.py --no-pred-cache` disables prewarm/reuse for a run. `kairos_pipeline.py --stage auto` uses the same module with a *separate ephemeral* tempdir — don't conflate the two.
- **Use `.has()`, not `.get()`, for existence checks.** `PredictionCache.get()` promotes a disk hit into the in-memory LRU as a side effect; calling it as a boolean probe (as `is_batch_cached()` once did) silently grows RAM until the machine freezes. `.has()` never touches `_mem`.
- **Model-major prewarm**: `kairos_papertrade.main()` runs `prewarm_prediction_cache()` before the date-major report loop so each model (base + each finetuned group) loads exactly once per run. `kairos_strategies.predict_all_batch` defers `_materialize_model` until *after* the shared-cache lookup and skips it on a full hit.
- **Long-run leak guards**: `_prediction_cache` (the per-process dict in `kairos_strategies.py`, distinct from `kairos_predcache`) is capped at 5000 entries (`_PREDICTION_CACHE_MAX_ENTRIES`) and cleared on overflow; the prewarm loops in `kairos_papertrade.py` call `gc.collect()` every 500 iterations (`_PREWARM_GC_INTERVAL`) because CPython's GC is allocation-count-based and won't collect few-but-large DataFrames on its own.
- **Crash mitigations**: `kairos_signals._connect_with_retry()` (3 attempts, backoff) for transient `sqlite3.OperationalError: unable to open database file`; `kairos_papertrade._raise_fd_limit()` (soft `RLIMIT_NOFILE` → hard cap, first line of `main()`) for `Errno 24 Too many open files`.
- **Report de-dup is persistent**: `generate_and_dedupe_reports()` keeps its `seen` map in a `SqliteDict` (`report_seen.db`, table `seen_<hash>` where the hash covers base_now/interval/work-item groups) so interrupted runs resume without regenerating reports. Two caveats: the filename is **CWD-relative** — run from the repo root or you get a fresh, empty DB — and accepted-finetuned model paths are **not** hashed, so a newly accepted finetuned model does not bust already-seen dates.
- **Per-strategy signals cache**: `kairos_signals.run()` caches each strategy's rows in the `signals_cache` table in `pipeline_results.db`, keyed by `(strategy, assets, interval, as_of_date, lookback, pred_samples, min_ev_pct, model_path, checkpoint_fingerprint)`. `as_of` is a *date*, not a timestamp, so overlapping backtest windows hit. Disabled strategies are never served stale (live registry check happens before any cache lookup). `--no-signal-cache` disables it (`use_signal_cache=True` is the default); writes are `INSERT OR REPLACE`, so growth is bounded by unique-key space.
- **Ops during report generation**: the shared `kairos.ops.GpuLock` is held for the whole `generate_and_dedupe_reports()` loop (other Kairos GPU jobs block meanwhile). Slow iterations (> `_SLOW_ITERATION_THRESHOLD_SECONDS`, 60s) trigger a Telegram heads-up plus a forensic snapshot (own PID/VmRSS, `free -h`, `nvidia-smi`) appended to `data/papertrade_watchdog.log`; per-group timing lines for slow or shared-cache-MISS groups go there too via `_log_group_timing`.

### Pipeline storage

The discovery pipeline persists results to:

- `data/pipeline_results.db` (SQLite, source of truth)
- `results/<stage>_<table>_<timestamp>.csv` (point-in-time mirrors)

Tables include `runs`, `universe_screen`, `correlation_pairs`, `suggested_groups`, `oracle_results`, `model_results`, `viability_report`, and `signals_cache` (per-strategy signals cache, see above).

---

## 8. Security Considerations

- **No secrets in source**: API keys, OAuth credentials, and exchange credentials must be passed via environment variables. `.env` is in `.gitignore`.
- **Google Sheets OAuth**: `strategy/credentials.json` and `strategy/token.json` are secrets and are `.gitignore`d. Do not commit them.
- **Database files**: `*.db`, `data/`, `results/`, `output/`, `finetune_csv/data`, `finetune_csv/models`, and `finetune_csv/train_data` are `.gitignore`d.
- **Model files**: `*.pth`, `*.pt`, `*.ckpt`, `*.bin` are `.gitignore`d.
- **Remote data**: `price_cache` can be configured to use a remote PostgreSQL store via `--remote` or `kairos.configure(remote=True)`. Keep connection strings out of committed code.

---

## 9. Useful Reference Files

| File | Purpose |
|------|---------|
| `README.md` | Project quickstart, component overview, model family, examples |
| `CLAUDE.md` | Project layout, run commands, hard-won gotchas |
| `strategy/README.md` | Strategy framework architecture, 42-strategy catalog, config reference |
| `strategy/PIPELINE.md` | Asset-discovery pipeline stages, DB schema, CLI reference |
| `docs/papertrade_tickets/` | Known papertrade gap-analysis tickets (exposure cap, costs, execution, sizing, model fitness, ...) |
| `docs/todo.md` | Epic/ticket tracker for active feature work |
| `ROADMAP.md` and `roadmap/*.md` | Long-term phase planning |
| `finetune_csv/README.md` | Custom CSV fine-tuning instructions |
| `webui/README.md` | Flask web UI usage |

---

## 10. TL;DR for Agents

- Use `uv run` for everything.
- Unit tests live in `tests/unit/` and must stay free of GPU/network/model dependencies.
- `strategy/` is not a package — add it to `sys.path` before importing strategy modules.
- Prefer raising typed `KairosError` subclasses over generic exceptions.
- Keep line length ≤ 120 and run flake8 / mypy before finishing non-trivial changes.
- Do not commit secrets, DBs, model weights, or generated `results/` / `output/` files.
