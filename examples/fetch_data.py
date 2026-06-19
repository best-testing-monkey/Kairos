"""
fetch_data.py

Warm (or inspect) the price_cache store for one or more A-share symbols.

This is the price_cache counterpart to examples/akshare/get_akshare_date_2024-2025_x.py.
That script used a multi-source fallback chain (akshare → baostock → EastMoney) to
download data and write it to a local CSV.  price_cache provides the same provider
chain internally and caches results in a local SQLite database, so the manual
CSV step is no longer required.

Usage:
    python fetch_data.py                   # fetch defaults (see STOCK_CODES below)
    python fetch_data.py --symbols 000001 600036
    python fetch_data.py --symbols 002594 --start 2024-01-01 --end 2025-12-31

The fetched data is cached automatically.  Subsequent calls for the same symbol /
interval / range are served from the local store with no network round-trip.
Use --export to also write a CSV alongside the cache (e.g. for inspection).
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append("../")
import price_cache

# ---- defaults ---------------------------------------------------------------
STOCK_CODES = ["002354", "600580", "300207", "300418", "000001", "600036"]
DEFAULT_START = "2024-01-01"
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")
DEFAULT_INTERVAL = "1d"
EXPORT_DIR = "./data"


def fetch_and_report(symbol: str, start: str, end: str, interval: str,
                     export: bool, export_dir: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Symbol: {symbol}  |  {start} → {end}  |  interval={interval}")
    print(f"{'=' * 60}")

    df = price_cache.get_price_data(symbol, start, end, interval=interval)

    if df is None or df.empty:
        print(f"  ERROR: no data returned for {symbol}")
        return

    # Basic summary
    print(f"  Bars fetched:   {len(df)}")
    print(f"  Date range:     {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Columns:        {list(df.columns)}")
    print(f"  Latest close:   {df['Close'].iloc[-1]:.2f}")

    # Per-year breakdown
    for year in sorted(df.index.year.unique()):
        yd = df[df.index.year == year]
        annual_return = (yd["Close"].iloc[-1] / yd["Close"].iloc[0] - 1) * 100
        print(f"  {year}: {len(yd):>3} bars  "
              f"hi={yd['High'].max():.2f}  lo={yd['Low'].min():.2f}  "
              f"return={annual_return:+.1f}%")

    if export:
        os.makedirs(export_dir, exist_ok=True)
        out_path = os.path.join(export_dir, f"{symbol}_stock_data.csv")
        df.reset_index().to_csv(out_path, index=False)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  Exported: {out_path}  ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Warm price_cache store and optionally export CSV")
    parser.add_argument("--symbols", nargs="+", default=STOCK_CODES,
                        help="Stock codes to fetch (default: built-in list)")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        help="Bar interval (1d, 1h, 5m, …)")
    parser.add_argument("--export", action="store_true",
                        help="Also write a CSV file per symbol")
    parser.add_argument("--export-dir", default=EXPORT_DIR,
                        help="Directory for exported CSVs")
    parser.add_argument("--db", default=None,
                        help="Override price_cache DB path")
    args = parser.parse_args()

    price_cache.configure(remote=False)
    if args.db:
        price_cache.DB_PATH = args.db

    print("price_cache data fetch utility")
    print(f"DB: {price_cache.DB_PATH}")
    print(f"Symbols: {args.symbols}")
    print(f"Range: {args.start} → {args.end}  interval={args.interval}")

    for symbol in args.symbols:
        try:
            fetch_and_report(symbol, args.start, args.end, args.interval,
                             args.export, args.export_dir)
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
