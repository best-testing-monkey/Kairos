"""
Fetch daily OHLCV data for a list of crypto tickers via price_cache
and save each as a CSV in train_data/crypto/.
"""
import sys
import os
import pandas as pd

sys.path.append('../')
import price_cache

TICKERS = [
    "BTC-USD",   # Bitcoin         - 2014
    "ETH-USD",   # Ethereum        - 2015
    "LTC-USD",   # Litecoin        - 2013
    "XRP-USD",   # Ripple          - 2013
    "DOGE-USD",  # Dogecoin        - 2013
    "XMR-USD",   # Monero          - 2014
    "DASH-USD",  # Dash            - 2014
    "XLM-USD",   # Stellar         - 2014
    "ETC-USD",   # Ethereum Classic- 2016
    "ZEC-USD",   # Zcash           - 2016
    "BCH-USD",   # Bitcoin Cash    - 2017
    "BNB-USD",   # Binance Coin    - 2017
    "ADA-USD",   # Cardano         - 2017
    "TRX-USD",   # TRON            - 2017
    "EOS-USD",   # EOS             - 2017
    "LINK-USD",  # Chainlink       - 2017
    "ATOM-USD",  # Cosmos          - 2019
    "MATIC-USD", # Polygon         - 2019
    "SOL-USD",   # Solana          - 2020
    "AVAX-USD",  # Avalanche       - 2020
    "DOT-USD",   # Polkadot        - 2020
    "UNI-USD",   # Uniswap         - 2020
]

OUT_DIR = "train_data/crypto"
START   = "2013-01-01"
# Stop before the backtest's 60-bar actual window (~last 60 daily bars from today).
# From run_backtest_kairos.py: actual = raw.iloc[-PRED_LEN:], context_end = raw.index[-(PRED_LEN+1)]
# With today ~2026-06-21 and PRED_LEN=60, actuals start ~2026-04-22. Cut off there.
END     = "2026-04-21"

price_cache.configure(remote=False)

for ticker in TICKERS:
    out_path = os.path.join(OUT_DIR, f"{ticker.replace('-', '_')}_1d.csv")
    print(f"Fetching {ticker} ...", end=" ", flush=True)
    try:
        raw = price_cache.get_price_data(ticker, START, END, interval="1d")
        if raw is None or raw.empty:
            print("NO DATA")
            continue

        raw = raw.sort_index().copy()
        raw.columns = [c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        if "amount" not in raw.columns or raw["amount"].isna().all():
            raw["amount"] = raw["close"] * raw["volume"]
        else:
            raw["amount"] = raw["amount"].astype("float64")

        raw = raw[["open", "high", "low", "close", "volume", "amount"]]
        raw.index.name = "timestamps"
        raw.to_csv(out_path)
        print(f"{len(raw)} rows  ({raw.index[0].date()} → {raw.index[-1].date()})")
    except Exception as e:
        print(f"ERROR: {e}")

print("\nDone.")
