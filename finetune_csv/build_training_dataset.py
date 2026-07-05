#!/usr/bin/env python3
"""
build_training_dataset.py

Fetches OHLCV data for a large basket of global instruments at daily (1d) and
hourly (1h) intervals and saves each as a separate CSV in train_data/.

The train_data/ directory is consumed by CustomKlineDataset when data_path
points to a directory - each CSV is an independent instrument and windows
never cross instrument boundaries.

Instruments:
  1d  ~240 instruments: US stocks, ETFs, CN A-shares, HK, Europe, Japan
  1h  ~45 liquid instruments (yfinance 1h only goes back 730 days)

Usage:
    cd finetune_csv
    uv run python build_training_dataset.py
    uv run python build_training_dataset.py --skip_hourly    # daily only
    uv run python build_training_dataset.py --workers 6      # more parallelism
"""

import os
import sys
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import price_cache

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'train_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY        = date.today().isoformat()
DAILY_START  = '1999-01-01'
HOURLY_START = (date.today() - timedelta(days=720)).isoformat()

# ── Instrument universe ───────────────────────────────────────────────────────

DAILY_INSTRUMENTS = [
    # --- US mega/large-cap ---
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "AVGO", "KO", "PEP", "COST", "WMT", "MCD", "BAC", "CRM",
    "TMO", "NFLX", "CSCO", "ACN", "ABT", "QCOM", "LIN", "WFC", "DHR",
    "TXN", "NEE", "RTX", "BMY", "MS", "AMGN", "IBM", "SPGI", "UNP",
    "PM", "GS", "LOW", "SBUX", "AXP", "ISRG", "BKNG", "SYK", "CI",
    "REGN", "ADP", "BLK", "MO", "GILD", "ZTS", "MMC", "T", "CB",
    "EOG", "SLB", "SO", "DUK", "CL", "NSC", "USB", "BSX", "AON", "ITW",
    "ADSK", "DIS", "INTC", "AMD", "MU", "AMAT", "KLAC", "LRCX", "SNPS",
    "CDNS", "MRVL", "ADI", "MCHP", "GLW", "F", "GM", "GE", "BA", "CAT",
    "DE", "MMM", "HON", "EMR", "ETN", "FDX", "UPS", "DAL", "AAL", "LUV",
    "CCL", "RCL", "MAR", "HLT", "CAH", "MCK",
    # --- US ETFs (diverse exposure) ---
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA",
    "EEM", "GLD", "SLV", "TLT", "HYG", "LQD", "XLF", "XLK", "XLE",
    "XLV", "XLI", "XLB", "XLU", "XLC", "XLRE", "XLP", "XLY", "VNQ",
    "ARKK", "IBIT", "VXX", "SQQQ", "TQQQ", "SDS", "SSO",
    # --- Chinese A-shares (.SZ = Shenzhen, .SS = Shanghai) ---
    "300750.SZ", "000001.SZ", "600519.SS", "000858.SZ", "601398.SS",
    "600036.SS", "002594.SZ", "300418.SZ", "600276.SS", "000333.SZ",
    "601318.SS", "600900.SS", "601166.SS", "000568.SZ", "600309.SS",
    "601628.SS", "000725.SZ", "601888.SS", "002415.SZ", "600887.SS",
    "600031.SS", "601012.SS", "000002.SZ", "601601.SS", "002352.SZ",
    "601919.SS", "600048.SS", "600104.SS", "002714.SZ", "000651.SZ",
    "600690.SS", "601766.SS", "000100.SZ", "603259.SS", "601138.SS",
    "600000.SS", "600016.SS", "601088.SS", "601225.SS", "600028.SS",
    "601857.SS", "601668.SS", "601390.SS", "601186.SS", "600585.SS",
    "601600.SS", "601111.SS", "601328.SS", "601336.SS", "000831.SZ",
    # --- Hong Kong ---
    "0700.HK", "9988.HK", "0939.HK", "1398.HK", "0941.HK",
    "2318.HK", "3690.HK", "0005.HK", "1299.HK", "2020.HK",
    "0388.HK", "0883.HK", "0857.HK", "2628.HK", "1109.HK",
    "0001.HK", "0002.HK", "0011.HK", "0016.HK", "0027.HK",
    "1177.HK", "0066.HK", "0688.HK", "0175.HK", "1038.HK",
    # --- European ---
    "SAP.DE", "SIE.DE", "ALV.DE", "BMW.DE", "VOW3.DE",
    "MC.PA", "TTE.PA", "SAN.PA", "BNP.PA", "AIR.PA",
    "SHEL.L", "BP.L", "HSBA.L", "RIO.L", "VOD.L",
    "ASML.AS", "NESN.SW", "NOVN.SW", "ROG.SW",
    # --- Japanese ---
    "7203.T", "6758.T", "9984.T", "8306.T", "7974.T",
    "6861.T", "6501.T", "9433.T", "8035.T", "4519.T",
]

HOURLY_INSTRUMENTS = [
    # US liquid
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "MA", "BAC", "GS", "MS", "WFC",
    "NFLX", "AMD", "INTC", "QCOM", "MU", "AMAT",
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLK",
    # CN / HK liquid
    "300750.SZ", "000001.SZ", "600519.SS", "002594.SZ", "300418.SZ",
    "000333.SZ", "000858.SZ", "600036.SS", "601166.SS",
    "0700.HK", "9988.HK", "3690.HK",
]

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _safe_filename(ticker: str) -> str:
    return ticker.replace('/', '_').replace('\\', '_').replace(':', '_')


def fetch_and_save(ticker: str, interval: str, start: str, end: str,
                   output_dir: str, retries: int = 3):
    fname    = f"{_safe_filename(ticker)}_{interval}.csv"
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        return ticker, 'cached', 0

    for attempt in range(retries):
        try:
            df = price_cache.get_price_data(ticker, start_date=start, end_date=end,
                                            interval=interval)
            if df is None or df.empty:
                return ticker, 'empty', 0

            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.reset_index()

            # Normalise the timestamp column produced by reset_index
            for candidate in ('datetime', 'date', 'index', 'Datetime', 'Date'):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'timestamps'})
                    break
            if 'timestamps' not in df.columns:
                df.columns = ['timestamps'] + list(df.columns[1:])

            required = ('open', 'high', 'low', 'close', 'volume')
            if not all(c in df.columns for c in required):
                return ticker, 'missing_cols', 0

            if 'amount' not in df.columns:
                df['amount'] = df['volume'] * df['close']

            df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            df = df.dropna(subset=['close'])

            if len(df) < 100:
                return ticker, f'too_short({len(df)})', len(df)

            df.to_csv(out_path, index=False)
            return ticker, 'ok', len(df)

        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ticker, f'error:{exc}', 0

    return ticker, 'failed', 0


def fetch_batch(instruments: list, interval: str, start: str, end: str,
                output_dir: str, workers: int = 4) -> dict:
    results = dict(ok=0, cached=0, failed=0, total_rows=0)

    price_cache.configure(remote=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_and_save, t, interval, start, end, output_dir): t
            for t in instruments
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            ticker, status, rows = fut.result()
            tag = f"[{done:3d}/{len(instruments)}]"
            if status == 'ok':
                results['ok']        += 1
                results['total_rows'] += rows
                print(f"  {tag} {ticker:<20s} ✓  {rows:6,} rows")
            elif status == 'cached':
                results['cached'] += 1
                print(f"  {tag} {ticker:<20s} (cached)")
            else:
                results['failed'] += 1
                print(f"  {tag} {ticker:<20s} ✗  {status}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',      type=int, default=4,
                        help='Parallel download workers (default: 4)')
    parser.add_argument('--skip_hourly', action='store_true',
                        help='Skip hourly data (daily only)')
    args = parser.parse_args()

    print(f"Output : {OUTPUT_DIR}")
    print(f"Daily  : {DAILY_START} → {TODAY}  ({len(DAILY_INSTRUMENTS)} instruments)")
    print(f"Hourly : {HOURLY_START} → {TODAY}  ({len(HOURLY_INSTRUMENTS)} instruments)")
    print()

    print(f"=== Daily (1d) instruments ===")
    r = fetch_batch(DAILY_INSTRUMENTS, '1d', DAILY_START, TODAY, OUTPUT_DIR, args.workers)
    print(f"  Fetched {r['ok']}, cached {r['cached']}, failed {r['failed']}, "
          f"new rows {r['total_rows']:,}\n")

    if not args.skip_hourly:
        print(f"=== Hourly (1h) instruments ===")
        r = fetch_batch(HOURLY_INSTRUMENTS, '1h', HOURLY_START, TODAY, OUTPUT_DIR, args.workers)
        print(f"  Fetched {r['ok']}, cached {r['cached']}, failed {r['failed']}, "
              f"new rows {r['total_rows']:,}\n")

    # Summary
    csv_files  = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
    total_rows = 0
    for f in csv_files:
        try:
            total_rows += sum(1 for _ in open(os.path.join(OUTPUT_DIR, f))) - 1
        except Exception:
            pass

    print("=== Dataset summary ===")
    print(f"  Instrument files : {len(csv_files)}")
    print(f"  Total rows       : {total_rows:,}")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print()
    print("Next step:")
    print("  Update data_path in configs/my_large_run.yaml to:")
    print(f"    {OUTPUT_DIR}")
    print("  Then re-run generate_distilled_tokens.py.")
