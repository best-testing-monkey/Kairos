# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Predicts future daily K-line (1D) data for A-share markets using Kronos and
price_cache.  Data is fetched and cached automatically — no manual CSV
wrangling required.

Usage:
    python prediction_cn_markets_day.py --symbol 000001.SS

Arguments:
    --symbol     Ticker symbol passed directly to price_cache.
                 The required format depends on which provider resolves it:
                   yfinance (default): use exchange suffixes — Shanghai stocks
                     end in .SS (e.g. 600580.SS), Shenzhen stocks in .SZ
                     (e.g. 000001.SZ, 002594.SZ).
                   akshare provider: bare 6-digit codes work (000001, 002594).
                 Check your price_cache installation to know which providers
                 are active.

Output:
    - Saves prediction results to ./outputs/pred_<symbol>_data.csv and
      ./outputs/pred_<symbol>_chart.png
    - Logs and progress are printed to console

Example:
    python prediction_cn_markets_day.py --symbol 000001.SZ   # Ping An Bank (yfinance)
    python prediction_cn_markets_day.py --symbol 002594.SZ   # BYD (yfinance)
"""

import os
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
import price_cache
from kairos.data import get_forecast_window
from model import Kronos, KronosTokenizer, KronosPredictor

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 400
PRED_LEN = 120
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1


def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    print(f"Applying ±{limit_rate*100:.0f}% price limit ...")
    pred_df = pred_df.reset_index(drop=True)
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)
        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                pred_df.at[i, col] = float(max(min(value, limit_up), limit_down))
        last_close = float(pred_df.at[i, "close"])

    return pred_df


def plot_result(x_df, x_timestamp, pred_df, y_timestamp, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(x_timestamp, x_df["close"], label="Historical", color="blue")
    plt.plot(y_timestamp, pred_df["close"], label="Predicted", color="red", linestyle="--")
    plt.title(f"Kronos Prediction for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_chart.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Chart saved: {plot_path}")


def predict_future(symbol):
    print(f"Loading Kronos tokenizer:{TOKENIZER_PRETRAINED} model:{MODEL_PRETRAINED} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
    model = Kronos.from_pretrained(MODEL_PRETRAINED)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    price_cache.configure(remote=False)

    print(f"Fetching {symbol} daily data via price_cache ...")
    x_df, x_timestamp, y_timestamp = get_forecast_window(
        symbol=symbol,
        interval="1d",
        lookback=LOOKBACK,
        pred_len=PRED_LEN,
        amount="auto",
    )
    print(f"Data loaded: {len(x_df)} rows, "
          f"range: {x_timestamp.iloc[0].date()} ~ {x_timestamp.iloc[-1].date()}")

    print("Generating predictions ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )

    last_close = float(x_df["close"].iloc[-1])
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # Attach dates and save
    pred_df["date"] = y_timestamp.values
    x_hist = x_df.copy()
    x_hist["date"] = x_timestamp.values

    df_out = pd.concat([
        x_hist[["date", "open", "high", "low", "close", "volume"]],
        pred_df[["date", "open", "high", "low", "close", "volume"]],
    ]).reset_index(drop=True)

    out_file = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Prediction completed and saved: {out_file}")

    plot_result(x_df, x_timestamp, pred_df, y_timestamp, symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos stock prediction script (price_cache)")
    parser.add_argument("--symbol", type=str, default="000001", help="Stock code")
    args = parser.parse_args()
    predict_future(symbol=args.symbol)
