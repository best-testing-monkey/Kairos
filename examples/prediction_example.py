"""
prediction_example.py

Basic Kronos stock prediction using price_cache for data fetching.

Usage:
    python prediction_example.py

Output:
    - ./output/<symbol>_prediction.png
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

SYMBOL   = "300418.SZ"   # Kunlun Wanwei — change to any yfinance ticker
LOOKBACK = 400
PRED_LEN = 120
OUTPUT_DIR = "./output"


def plot_prediction(x_df, x_timestamp, pred_df, y_timestamp, symbol, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax1.plot(x_timestamp.values[-200:], x_df["close"].values[-200:],
             label="Historical", color="steelblue", linewidth=1.5)
    ax1.plot(y_timestamp.values, pred_df["close"].values,
             label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Close Price")
    ax1.set_title(f"{symbol} — {PRED_LEN}-bar forecast")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_timestamp.values[-200:], x_df["volume"].values[-200:],
             label="Historical", color="steelblue", linewidth=1.5)
    ax2.plot(y_timestamp.values, pred_df["volume"].values,
             label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Volume")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{symbol.replace('.', '_')}_prediction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Chart saved: {path}")


if __name__ == "__main__":
    # 1. Fetch data
    price_cache.configure(remote=False)
    print(f"Fetching {SYMBOL} daily data via price_cache ...")
    x_df, x_timestamp, y_timestamp = get_forecast_window(
        symbol=SYMBOL,
        interval="1d",
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
        amount="auto",
    )
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
