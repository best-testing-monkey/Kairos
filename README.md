# Kairos

> Kairos couples [Kronos](https://github.com/shiyu-coder/Kronos), the open-source
> foundation model for the language of financial markets (K-lines), with
> [price_cache](https://github.com/best-testing-monkey/price_cache), a gap-aware,
> SQLite / PostgreSQL-backed OHLCV data layer with a seven-provider fallback chain.

Kronos consumes *Chronos*: sequential, recorded, quantitative time, the OHLCV
history a forecasting model is trained on. Kairos adds the other half of the
Greek pair: the *opportune moment*, the live present where data is pulled,
cached, shaped, and handed to the model with no manual CSV wrangling in between.

`price_cache` serves the historic record from a local cache (no network on a hit)
and fetches only genuinely missing ranges from its provider chain. Kairos is the
thin adapter that turns its output into exactly what `KronosPredictor.predict`
expects, then runs the forecast.

---

## What this fork adds

Kronos and price_cache do not speak the same dialect. Kairos is the translation
layer:

| Concern        | price_cache returns                                    | Kronos predict expects                          |
| -------------- | ------------------------------------------------------ | ----------------------------------------------- |
| Columns        | `Open, High, Low, Close, Volume, Dividends, ...`       | `open, high, low, close, volume[, amount]`      |
| Timestamps     | tz-aware `DatetimeIndex` (America/New_York)            | separate `x_timestamp` / `y_timestamp` Series   |
| Windowing      | date range (`start_date`, `end_date`)                  | bar counts (`lookback`, `pred_len`)             |
| Future bars    | not produced (it only returns observed history)        | `y_timestamp` Series for the periods to predict |

So Kairos provides one function that:

1. Pulls the lookback window via `price_cache.get_price_data`.
2. Renames OHLCV columns to the lowercase Kronos contract.
3. Lifts the `DatetimeIndex` into the `x_timestamp` Series.
4. Synthesizes `y_timestamp` by extrapolating the bar interval forward `pred_len` steps.

The result drops straight into the unmodified upstream `KronosPredictor`.

## Quickstart

```python
import price_cache
from kairos.data import get_forecast_window
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. Point price_cache at a store (local SQLite shown; remote/three-tier also supported)
price_cache.configure(remote=False)

# 2. Pull a Kronos-ready window: history + synthesized future timestamps
x_df, x_timestamp, y_timestamp = get_forecast_window(
    symbol="AAPL",
    interval="1h",        # any price_cache interval: 1m,5m,15m,1h,1d,1wk,...
    lookback=400,         # number of historic bars to feed the model
    pred_len=120,         # number of future bars to forecast
)

# 3. Standard, unmodified Kronos forecast
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=120,
    T=1.0,
    top_p=0.9,
    sample_count=1,
)
print(pred_df.head())
```

### Reference adapter

The adapter Kairos ships (`kairos/data.py`) is small; this is the shape of it.
Note the column map and the future-timestamp synthesis, which are the two pieces
Kronos cannot do on its own.

```python
import pandas as pd
import price_cache

# price_cache interval string -> pandas offset alias for extrapolating future bars.
# Daily and coarser should step on business days, not calendar days.
_FREQ = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "60min", "90m": "90min", "1h": "1h",
    "1d": "B", "5d": "5B", "1wk": "W", "1mo": "MS", "3mo": "QS",
}

_RENAME = {"Open": "open", "High": "high", "Low": "low",
           "Close": "close", "Volume": "volume"}

def get_forecast_window(symbol, interval, lookback, pred_len,
                        start_date=None, end_date=None):
    # Pull enough history to cover `lookback` bars. A real implementation sizes
    # the date range from interval * lookback; a wide range plus a tail slice
    # is the simplest correct version.
    df = price_cache.get_price_data(symbol, start_date, end_date, interval=interval)
    if df is None or len(df) < lookback:
        raise ValueError(f"insufficient cached/fetched data for {symbol} {interval}")

    df = df.tail(lookback)
    x_df = df.rename(columns=_RENAME)[["open", "high", "low", "close", "volume"]]
    # Kronos treats `amount` as optional and zero-fills it; include if you have it.

    x_timestamp = pd.Series(df.index)
    last = df.index[-1]
    freq = _FREQ[interval]
    future = pd.date_range(start=last, periods=pred_len + 1, freq=freq)[1:]
    y_timestamp = pd.Series(future)

    return x_df.reset_index(drop=True), x_timestamp, y_timestamp
```

For the full Kronos model usage, model zoo, and finetuning pipeline, see the
upstream documentation below.

---

## Examples

| File | Data source | Description |
|------|-------------|-------------|
| `examples/fetch_data.py` | price_cache | Warm the cache for one or more symbols; optionally export CSV |
| `examples/prediction_cn_markets_day.py` | price_cache | Daily A-share prediction with ±10 % price-limit clipping |
| `examples/prediction_new.py` | price_cache | Prediction with price + volume + change charts and CSV report |
| `examples/prediction_market_factors.py` | price_cache | Comprehensive prediction with multi-dimensional market-factor enhancement |
| `examples/prediction_GUI.py` | price_cache | Tkinter GUI wrapper around the price_cache workflow |
| `examples/prediction_example.py` | CSV | Basic Kronos usage from a local CSV |
| `examples/prediction_batch_example.py` | CSV | Batch prediction for multiple windows |
| `examples/prediction_wo_vol_example.py` | CSV | Prediction without volume data |
| `examples/run_backtest_kronos.py` | CSV | Backtesting framework |
| `examples/akshare/get_akshare_date_2024-2025_x.py` | akshare / EastMoney / Baostock | Multi-source data-fetch utility that writes a local CSV |
| `examples/akshare/prediction_cn_markets_day.py` | akshare | Daily A-share prediction fetching directly via akshare |
| `examples/akshare/prediction_new.py` | akshare + CSV | Two-step fetch-then-predict using akshare and a local CSV |
| `examples/akshare/prediction_akshare_2024-2025.py` | akshare | Comprehensive prediction with market-factor analysis via akshare |
| `examples/akshare/prediction_new_GUI.py` | akshare | Tkinter GUI wrapper around the akshare workflow |

The top-level `examples/` scripts use `price_cache` (and `kairos.data.get_forecast_window`) so
they work offline once the cache is warm and require no akshare dependency.  The
`examples/akshare/` subdirectory preserves the original scripts for users who prefer
fetching directly from akshare or need the richer Chinese-market metadata those scripts collect.

---

## Components

| Layer       | Project                                                             | Role                                                    |
| ----------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| Model       | [Kronos](https://github.com/shiyu-coder/Kronos)                    | K-line foundation model, tokenizer, predictor           |
| Data        | [price_cache](https://github.com/best-testing-monkey/price_cache)  | Gap-aware OHLCV cache, seven-provider fallback chain     |
| Glue        | Kairos (this repo)                                                 | Adapter from price_cache output to the Kronos contract  |

## Training a Kronos-Large Model

Kairos adds scripts to train a **Kronos-large** predictor (~499M parameters) on your own
data.  Two strategies are supported:

| Strategy | When to use |
|---|---|
| **Distillation warm-start → ground-truth finetune** | Recommended. The large model first learns from a smaller teacher (base/small), then refines on real data. Converges faster and often reaches lower loss. |
| **From-scratch on real data** | Simpler; set `training_mode: groundtruth_only`. Suitable when you have abundant data or don't have a finetuned teacher available. |

### Model family

| Model | Params | d_model | n_layers | n_heads | ff_dim |
|-------|--------|---------|----------|---------|--------|
| Kronos-mini  |   4.1M |  ~128  |   6  |  8  |  512  |
| Kronos-small |  24.7M |  ~384  |  12  | 12  | 1024  |
| Kronos-base  | 102.3M |   832  |  12  | 16  | 2048  |
| **Kronos-large** | **~499M** | **1536** | **22** | **24** | **4096** |

### Prerequisites

```bash
# Same dependencies as the existing finetuning pipeline
pip install torch pyyaml pandas numpy
```

You also need a **finetuned tokenizer** from an earlier `finetune_csv` run and a **teacher
predictor** checkpoint (e.g. a finetuned `Kronos-base`).

### Step 1 — Edit the config

Copy and fill in the template:

```bash
cp finetune_csv/configs/config_kronos_large.yaml finetune_csv/configs/my_large_run.yaml
```

Key fields:

```yaml
data:
  data_path: "/path/to/your/data.csv"   # same CSV used for finetuning

model_paths:
  finetuned_tokenizer: "/path/to/finetuned/tokenizer/best_model"
  teacher_predictor:   "/path/to/finetuned/basemodel/best_model"
  distill_cache_dir:   "/path/to/distill_cache"    # will be created
  base_path:           "/path/to/save/output"
  exp_name:            "my_large_run"

experiment:
  # 'distill_then_finetune' | 'distill_only' | 'groundtruth_only'
  training_mode: "distill_then_finetune"
```

See `finetune_csv/configs/config_kronos_large.yaml` for all options with inline comments.

### Step 2 — Generate distillation token cache

This one-time step runs the teacher model over your training data and saves both the
teacher-predicted tokens and ground-truth tokens to disk.  Training never reloads the
teacher, so GPU memory is fully available to the large model.

```bash
cd finetune_csv
python generate_distilled_tokens.py --config configs/my_large_run.yaml
# generates: distill_cache/train_distilled_tokens.pt
#            distill_cache/val_distilled_tokens.pt
```

Options:

```
--split train,val   Which splits to generate (default: train,val)
--sample            Use multinomial sampling instead of argmax for teacher predictions
```

> **Skip this step** if using `training_mode: groundtruth_only` — a GT-only cache can be
> generated with `--split train,val` using any Kronos model as the "teacher" (its predicted
> tokens will be ignored; only the ground-truth tokens are used in Phase 2).

### Step 3 — Train

```bash
python train_large_model.py --config configs/my_large_run.yaml
```

With `distill_then_finetune` (default) the trainer runs sequentially:

```
Phase 1 — Distillation warm-start
  └─ trains on teacher-predicted tokens
  └─ saves best checkpoint to: <base_path>/<exp_name>/kronos_large/phase1_best/best_model/

Phase 2 — Ground-truth finetune
  └─ trains on real tokenizer outputs
  └─ saves best checkpoint to: <base_path>/<exp_name>/kronos_large/phase2_best/best_model/
```

Optional flags:

```
--skip-phase1   Jump straight to Phase 2 (useful for resuming)
--skip-phase2   Run Phase 1 only
```

For from-scratch training (no teacher needed):

```yaml
# in config:
experiment:
  training_mode: "groundtruth_only"
```

```bash
python train_large_model.py --config configs/my_large_run.yaml
```

### Config reference

```yaml
large_model_arch:         # Kronos-large architecture (~499M params)
  d_model: 1536
  n_layers: 22
  n_heads: 24             # head_dim = 1536 / 24 = 64
  ff_dim: 4096
  ffn_dropout_p: 0.1      # lower dropout than base (0.2) due to larger capacity
  resid_dropout_p: 0.1
  s1_bits: 10             # must match the tokenizer
  s2_bits: 10

training:
  phase1_epochs: 15       # distillation warm-start epochs
  phase1_batch_size: 16
  phase1_learning_rate: 1.0e-4
  phase1_grad_clip: 3.0

  phase2_epochs: 30       # ground-truth finetune epochs
  phase2_batch_size: 8
  phase2_learning_rate: 2.0e-5   # lower LR avoids forgetting Phase 1 gains
  phase2_grad_clip: 1.0

  accumulation_steps: 4   # effective batch = batch_size × accumulation_steps
  distill_batch_size: 32  # batch size used when generating the cache

distillation:
  sample_mode: "argmax"   # 'argmax' (stable) or 'sample' (diverse)
  sampling_temperature: 1.0
  top_p: 1.0
```

### Output structure

```
<base_path>/<exp_name>/
  kronos_large/
    phase1_best/best_model/    ← best Phase 1 checkpoint (distillation)
    phase2_best/best_model/    ← best Phase 2 checkpoint (final model)
  logs/
    large_model_training_rank_0.log
```

### Inference with the trained model

Load the final checkpoint exactly like any other Kronos model:

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("/path/to/finetuned/tokenizer/best_model")
model = Kronos.from_pretrained(
    "/path/to/finetuned/kronos_large/phase2_best/best_model"
)
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Then use exactly as you would with Kronos-base:
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=48,
    T=1.0,
    top_p=0.9,
    sample_count=5,
)
```

---

## Upstream: Kronos

This project is a fork of **Kronos: A Foundation Model for the Language of
Financial Markets** by Yu Shi et al. The model, tokenizer, predictor, and
finetuning code are their work.

- Repository: https://github.com/shiyu-coder/Kronos
- Paper: https://arxiv.org/abs/2508.02739

```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```

## License

MIT, inherited from upstream Kronos. price_cache is likewise MIT. See
[LICENSE](LICENSE); original copyright notices are retained.
