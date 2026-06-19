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
