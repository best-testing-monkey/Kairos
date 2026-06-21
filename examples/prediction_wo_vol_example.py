"""
prediction_wo_vol_example.py

Kronos stock prediction using only OHLC columns (no volume/amount).
Data is fetched via price_cache — no local CSV file required.

Usage:
    python prediction_wo_vol_example.py

Output:
    - ./output/<symbol>_prediction_wo_vol.png
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
import price_cache
from kairos.data import get_forecast_window
from model import Kronos, KronosTokenizer, KronosPredictor

SYMBOL     = "300418.SZ"   # Kunlun Wanwei — change to any yfinance ticker
LOOKBACK   = 400
PRED_LEN   = 120
OUTPUT_DIR = "./output"


def plot_prediction(x_df, x_timestamp, pred_df, y_timestamp, symbol, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_timestamp.values[-200:], x_df["close"].values[-200:],
            label="Historical", color="steelblue", linewidth=1.5)
    ax.plot(y_timestamp.values, pred_df["close"].values,
            label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
    ax.set_ylabel("Close Price")
    ax.set_title(f"{symbol} — {PRED_LEN}-bar forecast (close only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{symbol.replace('.', '_')}_prediction_wo_vol.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Chart saved: {path}")


if __name__ == "__main__":
    # 1. Fetch data — keep only OHLC (drop volume/amount)
    price_cache.configure(remote=False)
    print(f"Fetching {SYMBOL} daily data via price_cache ...")
    x_df, x_timestamp, y_timestamp = get_forecast_window(
        symbol=SYMBOL,
        interval="1d",
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
        amount="auto",
    )
    x_df = x_df[["open", "high", "low", "close"]]
    print(f"Loaded {len(x_df)} bars  "
          f"({x_timestamp.iloc[0].date()} → {x_timestamp.iloc[-1].date()})")

    # 2. Load model
    print("Loading Kronos model ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 3. Predict
    print("Running prediction ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    # 4. Results
    print("\nForecasted Data Head:")
    print(pred_df.head())
    plot_prediction(x_df, x_timestamp, pred_df, y_timestamp, SYMBOL, OUTPUT_DIR)
    print("Done.")
