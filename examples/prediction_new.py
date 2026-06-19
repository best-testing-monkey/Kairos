"""
prediction_new.py

Comprehensive Kronos stock prediction using price_cache for data fetching.
Replaces the two-step akshare-fetch + CSV-load workflow with a single
get_forecast_window() call backed by price_cache's seven-provider chain.

Usage:
    python prediction_new.py

Modify STOCK_CONFIG at the bottom to target a different symbol.

Output:
    - <symbol>_prediction_chart.png  — price + volume + change chart
    - <symbol>_detailed_predictions.csv — per-bar forecast data
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.append("../")
import price_cache
from kairos.data import get_forecast_window

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("WARNING: Cannot import Kronos model; prediction functionality unavailable")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_prediction_with_details(x_df, x_timestamp, pred_df, y_timestamp,
                                 stock_code, stock_name, pred_len, output_dir):
    ensure_output_directory(output_dir)

    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    hist_close = pd.Series(x_df["close"].values, index=x_timestamp)
    pred_close = pd.Series(pred_df["close"].values, index=y_timestamp)

    ax1.plot(hist_close.index[-200:], hist_close.values[-200:],
             label="Historical Price", color="#1f77b4", linewidth=2.5)
    ax1.plot(pred_close.index, pred_close.values,
             label="Predicted Price", color="#ff7f0e", linewidth=2.5,
             linestyle="-", marker="o", markersize=3)
    ax1.axvline(x=y_timestamp.iloc[0], color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.set_ylabel("Close Price (CNY)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{stock_name}({stock_code}) — Next {pred_len} Trading Days",
                  fontsize=16, fontweight="bold", pad=20)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    pred_vol = pd.Series(pred_df["volume"].values, index=y_timestamp)
    ax2.bar(pred_vol.index, pred_vol.values, alpha=0.7, color="#ff7f0e",
            label="Predicted Volume", width=0.8)
    ax2.set_ylabel("Volume (lots)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True, alpha=0.3)

    last_hist_close = float(hist_close.iloc[-1])
    price_change = pred_close - last_hist_close
    bar_colors = ["green" if x >= 0 else "red" for x in price_change]
    ax3.bar(range(len(price_change)), price_change, alpha=0.8, color=bar_colors)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
    ax3.set_ylabel("Price Change (CNY)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Trading Day", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"{stock_code}_prediction_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Prediction chart saved: {chart_path}")
    plt.show()

    return hist_close, pred_close


def generate_prediction_report(hist_close, pred_close, y_timestamp,
                                stock_code, stock_name, output_dir):
    ensure_output_directory(output_dir)
    print(f"\n{'=' * 70}")
    print(f"  {stock_name}({stock_code}) Stock Prediction Report")
    print(f"{'=' * 70}")

    current = float(hist_close.iloc[-1])
    final = float(pred_close.iloc[-1])
    change_pct = (final / current - 1) * 100

    print(f"  Current Price:      {current:.2f} CNY")
    print(f"  Predicted End:      {final:.2f} CNY  ({change_pct:+.2f}%)")
    print(f"  Predicted High:     {pred_close.max():.2f} CNY")
    print(f"  Predicted Low:      {pred_close.min():.2f} CNY")
    print(f"  Predicted Avg:      {pred_close.mean():.2f} CNY")
    print(f"  Prediction Period:  {len(pred_close)} bars  "
          f"({y_timestamp.iloc[0].date()} → {y_timestamp.iloc[-1].date()})")

    details = pd.DataFrame({
        "date": y_timestamp.values,
        "predicted_close": pred_close.values,
        "price_change_cny": (pred_close.values - current),
        "price_change_pct": ((pred_close.values / current - 1) * 100),
    })
    out_file = os.path.join(output_dir, f"{stock_code}_detailed_predictions.csv")
    details.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")


def run_prediction(stock_code, stock_name, pred_days, output_dir,
                   lookback=300, device="cpu"):
    price_cache.configure(remote=False)

    print(f"Fetching {stock_name}({stock_code}) daily data via price_cache ...")
    x_df, x_timestamp, y_timestamp = get_forecast_window(
        symbol=stock_code,
        interval="1d",
        lookback=lookback,
        pred_len=pred_days,
        amount="auto",
    )
    print(f"Loaded {len(x_df)} bars  "
          f"({x_timestamp.iloc[0].date()} → {x_timestamp.iloc[-1].date()})")

    print("Loading Kronos model ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    print("Running prediction ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_days,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    hist_close, pred_close = plot_prediction_with_details(
        x_df, x_timestamp, pred_df, y_timestamp,
        stock_code, stock_name, pred_days, output_dir,
    )
    generate_prediction_report(
        hist_close, pred_close, y_timestamp,
        stock_code, stock_name, output_dir,
    )
    print(f"\nDone: {stock_name}({stock_code})")


if __name__ == "__main__":
    STOCK_CONFIG = {
        "stock_code": "300418",
        "stock_name": "Kunlun Wanwei",
        "pred_days": 100,
        "output_dir": "./output",
        "lookback": 300,
        "device": "cpu",
    }

    # Other examples:
    # STOCK_CONFIG = {"stock_code": "000001", "stock_name": "Ping An Bank", ...}
    # STOCK_CONFIG = {"stock_code": "600036", "stock_name": "China Merchants Bank", ...}
    # STOCK_CONFIG = {"stock_code": "300750", "stock_name": "CATL", ...}

    run_prediction(**STOCK_CONFIG)
