"""
prediction_batch_example.py

Demonstrates Kronos batch prediction across multiple symbols.  Data is fetched
live via price_cache — no local CSV file required.

Usage:
    python prediction_batch_example.py

Output:
    - ./output/<symbol>_batch_pred.png  — close-price chart per symbol
    - ./output/batch_predictions.csv    — all predictions in one table
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

# ── Configuration ────────────────────────────────────────────────────────────
SYMBOLS = [
    ("000001.SZ", "Ping An Bank"),
    ("300418.SZ", "Kunlun Wanwei"),
    ("600519.SS", "Kweichow Moutai"),
    ("002594.SZ", "BYD"),
    ("600036.SS", "China Merchants Bank"),
]
LOOKBACK = 300
PRED_LEN = 30
OUTPUT_DIR = "./output"
# ─────────────────────────────────────────────────────────────────────────────


def fetch_windows(symbols, lookback, pred_len):
    """Return parallel lists ready for predict_batch."""
    price_cache.configure(remote=False)
    dfs, x_timestamps, y_timestamps, labels = [], [], [], []
    for symbol, name in symbols:
        print(f"  Fetching {name} ({symbol}) ...")
        try:
            x_df, x_ts, y_ts = get_forecast_window(
                symbol=symbol,
                interval="1d",
                lookback=lookback,
                pred_len=pred_len,
                amount="auto",
            )
            dfs.append(x_df)
            x_timestamps.append(x_ts)
            y_timestamps.append(y_ts)
            labels.append((symbol, name))
            print(f"    {len(x_df)} bars  "
                  f"({x_ts.iloc[0].date()} → {x_ts.iloc[-1].date()})")
        except Exception as exc:
            print(f"    Skipped — {exc}")
    return dfs, x_timestamps, y_timestamps, labels


def save_chart(x_df, x_ts, pred_df, y_ts, symbol, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax1.plot(x_ts.values[-100:], x_df["close"].values[-100:],
             label="Historical", color="steelblue", linewidth=1.5)
    ax1.plot(y_ts.values, pred_df["close"].values,
             label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
    ax1.set_title(f"{name} ({symbol}) — {len(y_ts)}-day forecast",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(pred_df)), pred_df["volume"].values,
            color="steelblue", alpha=0.7)
    ax2.set_ylabel("Predicted Volume")
    ax2.set_xlabel("Trading Day")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{symbol.replace('.', '_')}_batch_pred.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Chart saved: {path}")


if __name__ == "__main__":
    print("Step 1: Fetching data ...")
    dfs, x_timestamps, y_timestamps, labels = fetch_windows(SYMBOLS, LOOKBACK, PRED_LEN)

    if not dfs:
        print("No data could be fetched. Check your network connection.")
        raise SystemExit(1)

    print(f"\nStep 2: Loading Kronos model ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    print(f"\nStep 3: Running batch prediction over {len(dfs)} symbols ...")
    pred_list = predictor.predict_batch(
        df_list=dfs,
        x_timestamp_list=x_timestamps,
        y_timestamp_list=y_timestamps,
        pred_len=PRED_LEN,
    )

    print("\nStep 4: Saving charts and combined CSV ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    for (symbol, name), x_df, x_ts, pred_df, y_ts in zip(
            labels, dfs, x_timestamps, pred_list, y_timestamps):
        save_chart(x_df, x_ts, pred_df, y_ts, symbol, name, OUTPUT_DIR)
        for date, row in zip(y_ts, pred_df.itertuples(index=False)):
            all_rows.append({
                "symbol": symbol,
                "name": name,
                "date": date,
                "predicted_close": row.close,
                "predicted_volume": row.volume,
            })

    combined = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "batch_predictions.csv")
    combined.to_csv(csv_path, index=False)
    print(f"Combined CSV saved: {csv_path}")

    print("\n=== Summary ===")
    for (symbol, name), pred_df, x_df, x_ts in zip(labels, pred_list, dfs, x_timestamps):
        current = float(x_df["close"].iloc[-1])
        end_pred = float(pred_df["close"].iloc[-1])
        change_pct = (end_pred / current - 1) * 100
        print(f"  {name:25s}  current={current:8.2f}  "
              f"pred_end={end_pred:8.2f}  ({change_pct:+.1f}%)")

    print("\nDone.")
