"""
prediction_GUI.py

Tkinter GUI for Kronos stock prediction using price_cache for data.

This is the price_cache counterpart to examples/akshare/prediction_new_GUI.py.
The UI layout and threading model are identical; only the data layer changes —
get_forecast_window() replaces the akshare fetch + CSV-read workflow.

Usage:
    python prediction_GUI.py
"""

import os
import sys
import threading
import warnings
from datetime import datetime

import matplotlib
# Use TkAgg when running inside the GUI event loop; fall back to Agg for
# headless environments (tests, CI) where no display is available.
import os as _os
if _os.environ.get("DISPLAY") or sys.platform == "win32" or sys.platform == "darwin":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox, ttk

warnings.filterwarnings("ignore")

sys.path.append("../")
import price_cache
from kairos.data import get_forecast_window

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("WARNING: Cannot import Kronos model; prediction unavailable")


# ==================== GUI ====================

class StockPredictorGUI:
    """Kronos stock prediction GUI backed by price_cache."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Kronos Stock Prediction System (price_cache)")
        self.root.geometry("820x640")
        self.root.configure(bg="#f0f0f0")
        self._create_widgets()

    def _create_widgets(self):
        tk.Label(self.root, text="Kronos Stock Prediction System",
                 font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#2c3e50").pack(pady=10)
        tk.Label(self.root,
                 text="Multi-dimensional prediction via price_cache + Kronos",
                 font=("Arial", 10), bg="#f0f0f0", fg="#7f8c8d").pack(pady=4)

        main = tk.Frame(self.root, bg="#f0f0f0")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # --- input section ---
        inp = tk.LabelFrame(main, text="Stock Parameters",
                            font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#2c3e50")
        inp.pack(fill=tk.X, pady=8)

        def _lbl(parent, text, row, col):
            tk.Label(parent, text=text, bg="#f0f0f0",
                     font=("Arial", 10)).grid(row=row, column=col, sticky=tk.W, padx=6, pady=5)

        def _entry(parent, var, row, col, width=15):
            e = tk.Entry(parent, textvariable=var, font=("Arial", 10), width=width)
            e.grid(row=row, column=col, padx=6, pady=5)
            return e

        # Default uses yfinance .SS suffix; change to bare code if akshare provider is active
        self.symbol_var = tk.StringVar(value="600580.SS")
        self.name_var = tk.StringVar(value="Wolong Electric Drive")
        self.pred_days_var = tk.StringVar(value="60")
        self.lookback_var = tk.StringVar(value="300")
        self.interval_var = tk.StringVar(value="1d")

        _lbl(inp, "Symbol:", 0, 0);  _entry(inp, self.symbol_var, 0, 1)
        _lbl(inp, "Name:", 0, 2);    _entry(inp, self.name_var, 0, 3)
        _lbl(inp, "Pred Days:", 1, 0); _entry(inp, self.pred_days_var, 1, 1)
        _lbl(inp, "Lookback:", 1, 2);  _entry(inp, self.lookback_var, 1, 3)
        _lbl(inp, "Interval:", 2, 0);  _entry(inp, self.interval_var, 2, 1)

        # --- output dir ---
        out_frame = tk.LabelFrame(main, text="Output Directory",
                                  font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#2c3e50")
        out_frame.pack(fill=tk.X, pady=8)
        self.output_dir_var = tk.StringVar(value="./output")
        tk.Entry(out_frame, textvariable=self.output_dir_var,
                 font=("Arial", 10), width=50).grid(row=0, column=0, padx=6, pady=5)
        tk.Button(out_frame, text="Browse",
                  command=self._browse_output).grid(row=0, column=1, padx=6)

        # --- buttons ---
        btn_frame = tk.Frame(main, bg="#f0f0f0")
        btn_frame.pack(pady=16)

        self.predict_btn = tk.Button(
            btn_frame, text="Start Prediction", command=self._start,
            font=("Arial", 12, "bold"), bg="#3498db", fg="white", width=16, height=2)
        self.predict_btn.pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="Reset", command=self._reset,
                  font=("Arial", 10), bg="#95a5a6", fg="white",
                  width=10, height=2).pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="Exit", command=self.root.quit,
                  font=("Arial", 10), bg="#e74c3c", fg="white",
                  width=10, height=2).pack(side=tk.LEFT, padx=8)

        # --- progress ---
        prog_frame = tk.LabelFrame(main, text="Progress",
                                   font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#2c3e50")
        prog_frame.pack(fill=tk.X, pady=8)
        self.progress_var = tk.StringVar(value="Waiting to start ...")
        tk.Label(prog_frame, textvariable=self.progress_var, bg="#f0f0f0",
                 font=("Arial", 10), wraplength=720, justify=tk.LEFT).pack(padx=10, pady=8, fill=tk.X)
        self.progress_bar = ttk.Progressbar(prog_frame, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=4)

        # --- results ---
        res_frame = tk.LabelFrame(main, text="Results",
                                  font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#2c3e50")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        self.result_text = tk.Text(res_frame, height=8, font=("Arial", 9), wrap=tk.WORD)
        sb = tk.Scrollbar(res_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=sb.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    # --- helpers ---

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir_var.set(d)

    def _reset(self):
        self.symbol_var.set("600580")
        self.name_var.set("Wolong Electric Drive")
        self.pred_days_var.set("60")
        self.lookback_var.set("300")
        self.interval_var.set("1d")
        self.output_dir_var.set("./output")
        self.result_text.delete(1.0, tk.END)
        self.progress_var.set("Waiting to start ...")

    def _update_progress(self, msg: str):
        self.root.after(0, lambda: self.progress_var.set(msg))
        print(msg)

    def _append_result(self, msg: str):
        self.root.after(0, lambda: self.result_text.insert(tk.END, msg + "\n"))
        self.root.after(0, lambda: self.result_text.see(tk.END))

    def _validate(self) -> bool:
        try:
            if not self.symbol_var.get().strip():
                messagebox.showerror("Error", "Please enter a symbol")
                return False
            pred_days = int(self.pred_days_var.get())
            if not (1 <= pred_days <= 365):
                messagebox.showerror("Error", "Prediction days must be 1–365")
                return False
            lookback = int(self.lookback_var.get())
            if not (50 <= lookback <= 2000):
                messagebox.showerror("Error", "Lookback must be 50–2000")
                return False
            return True
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return False

    def _start(self):
        if not self._validate():
            return
        self.predict_btn.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)
        self.progress_bar.start()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        try:
            symbol = self.symbol_var.get().strip()
            name = self.name_var.get().strip()
            pred_days = int(self.pred_days_var.get())
            lookback = int(self.lookback_var.get())
            interval = self.interval_var.get().strip()
            output_dir = self.output_dir_var.get()

            run_prediction_gui(
                symbol=symbol,
                stock_name=name,
                pred_days=pred_days,
                lookback=lookback,
                interval=interval,
                output_dir=output_dir,
                progress_cb=self._update_progress,
                result_cb=self._append_result,
            )
            self._update_progress("Prediction complete!")
            messagebox.showinfo("Done", f"{name}({symbol}) prediction complete!")
        except Exception as exc:
            self._update_progress(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))
        finally:
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
            self.root.after(0, self.progress_bar.stop)


# ==================== Prediction backend ====================

def run_prediction_gui(symbol, stock_name, pred_days, lookback, interval,
                       output_dir, progress_cb=print, result_cb=print):
    os.makedirs(output_dir, exist_ok=True)

    progress_cb(f"Fetching {symbol} {interval} data via price_cache ...")
    price_cache.configure(remote=False)
    x_df, x_timestamp, y_timestamp = get_forecast_window(
        symbol=symbol,
        interval=interval,
        lookback=lookback,
        pred_len=pred_days,
        amount="auto",
    )
    progress_cb(f"Loaded {len(x_df)} bars "
                f"({x_timestamp.iloc[0].date()} → {x_timestamp.iloc[-1].date()})")

    progress_cb("Loading Kronos model ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    progress_cb("Running prediction ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_days,
        T=1.0,
        top_p=0.9,
        sample_count=1,
    )

    progress_cb("Generating chart ...")
    _plot_and_save(x_df, x_timestamp, pred_df, y_timestamp,
                   symbol, stock_name, output_dir)

    current = float(x_df["close"].iloc[-1])
    final = float(pred_df["close"].iloc[-1])
    change_pct = (final / current - 1) * 100

    result_cb(f"Current price:  {current:.2f}")
    result_cb(f"Predicted end:  {final:.2f}  ({change_pct:+.2f}%)")
    result_cb(f"Predicted high: {pred_df['close'].max():.2f}")
    result_cb(f"Predicted low:  {pred_df['close'].min():.2f}")
    result_cb(f"Period:         {y_timestamp.iloc[0].date()} → {y_timestamp.iloc[-1].date()}")


def _plot_and_save(x_df, x_timestamp, pred_df, y_timestamp,
                   symbol, stock_name, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    hist_close = pd.Series(x_df["close"].values, index=x_timestamp)
    pred_close = pd.Series(pred_df["close"].values, index=y_timestamp)

    ax1.plot(hist_close.index[-200:], hist_close.values[-200:],
             color="#1f77b4", linewidth=2, label="Historical")
    ax1.plot(pred_close.index, pred_close.values,
             color="#ff7f0e", linewidth=2, linestyle="-",
             marker="o", markersize=3, label="Predicted")
    ax1.axvline(x=y_timestamp.iloc[0], color="red", linestyle="--", alpha=0.6)
    ax1.set_title(f"{stock_name}({symbol}) — Kronos Prediction", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    last = float(hist_close.iloc[-1])
    changes = pred_close - last
    ax2.bar(range(len(changes)), changes,
            color=["green" if v >= 0 else "red" for v in changes], alpha=0.8)
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Δ Close")
    ax2.set_xlabel("Trading Day")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{symbol}_prediction_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {path}")
    plt.close(fig)


# ==================== Entry point ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorGUI(root)
    root.mainloop()
