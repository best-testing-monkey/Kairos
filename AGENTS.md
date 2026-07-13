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

Key external Python dependency: `price_cache` installed directly from Git (`git+https://github.com/best-testing-monkey/price_cache.git`).

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
├── scripts/                # gpu_recover.py, smoke.py
├── data/                   # SQLite databases (ignored by git)
├── results/                # Pipeline CSV reports (ignored by git)
├── output/                 # Example / report outputs (ignored by git)
├── docs/                   # RFCs, tickets, todo
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
```

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

### Pipeline storage

The discovery pipeline persists results to:

- `data/pipeline_results.db` (SQLite, source of truth)
- `results/<stage>_<table>_<timestamp>.csv` (point-in-time mirrors)

Tables include `runs`, `universe_screen`, `correlation_pairs`, `suggested_groups`, `oracle_results`, `model_results`, and `viability_report`.

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
