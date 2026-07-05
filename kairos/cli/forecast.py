"""kairos forecast - fetch live/historic price data and run a Kronos prediction.

Usage:
    uv run forecast --model kronos-mini --symbol AAPL
    uv run forecast --model kronos-small --symbol MSFT --lookback 128 --pred-len 8
    uv run forecast --model kronos-base  --symbol TSLA --end 2024-06-14 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forecast",
        description="Fetch OHLCV data via price_cache and run a Kronos forecast.",
    )
    p.add_argument("--model", required=True,
                   help="Model name (kronos-mini | kronos-small | kronos-base) or HF repo ID.")
    p.add_argument("--symbol", required=True,
                   help="Ticker symbol, e.g. AAPL.")
    p.add_argument("--interval", default="1d",
                   help="Bar interval (default: 1d). Supported: 1m 5m 15m 30m 1h 1d 1wk …")
    p.add_argument("--lookback", type=int, default=64,
                   help="Number of historical bars to feed the model (default: 64).")
    p.add_argument("--pred-len", type=int, default=8,
                   help="Number of future bars to predict (default: 8).")
    p.add_argument("--end", default=None,
                   help="End date YYYY-MM-DD (default: today / live).")
    p.add_argument("--device", default="cpu",
                   help="Torch device: cpu | cuda | cuda:0 (default: cpu).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (default: 1.0).")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="Nucleus sampling top-p (default: 0.9).")
    p.add_argument("--sample-count", type=int, default=1,
                   help="Number of sample trajectories (default: 1).")
    p.add_argument("--calendar", default=None,
                   help="exchange_calendars code (default: XNYS).")
    p.add_argument("--output", choices=["table", "json", "csv"], default="table",
                   help="Output format (default: table).")
    p.add_argument("--remote", action="store_true",
                   help="Use remote PostgreSQL price_cache instead of local SQLite.")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ setup
    import kairos

    kairos.configure(
        remote=args.remote,
        calendar=args.calendar or "XNYS",
    )

    # ------------------------------------------------------------ fetch window
    print(f"Fetching {args.lookback} bars of {args.symbol} ({args.interval}) …")
    x_df, x_ts, y_ts = kairos.get_forecast_window(
        symbol=args.symbol,
        interval=args.interval,
        lookback=args.lookback,
        pred_len=args.pred_len,
        end=args.end,
        amount="auto",
        calendar=args.calendar,
    )

    print(f"Context window: {x_ts.iloc[0].date()} → {x_ts.iloc[-1].date()}  ({len(x_df)} bars)")
    print(f"Predicting:     {y_ts.iloc[0].date()} → {y_ts.iloc[-1].date()}  ({len(y_ts)} bars)")

    # --------------------------------------------------------------- load model
    from kairos.cli._models import load_predictor
    predictor = load_predictor(args.model, device=args.device)

    # ----------------------------------------------------------------- predict
    print("Running forecast …")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=args.pred_len,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=False,
    )
    pred_df.index = y_ts.values

    # ------------------------------------------------------------------ output
    _print_output(pred_df, args.output)


def _print_output(pred_df, fmt: str) -> None:
    if fmt == "json":
        records = pred_df.reset_index().rename(columns={"index": "timestamp"})
        records["timestamp"] = records["timestamp"].astype(str)
        print(json.dumps(records.to_dict(orient="records"), indent=2))
    elif fmt == "csv":
        print(pred_df.to_csv())
    else:
        print("\nForecast:")
        print(pred_df.to_string())


if __name__ == "__main__":
    main()
