#!/usr/bin/env python3
"""Time cost of auto_regressive_inference (model/kronos.py) per additional
predicted bar (pred_len), at the real 3-asset x 100-sample batch shape.

auto_regressive_inference already supports pred_len > 1 -- it loops one
autoregressive step per bar (model/kronos.py:423), reusing the once-only
context encode for step 0 and paying a full decode_s1+decode_s2 pass per
extra bar. This script measures what that loop actually costs on this GPU.

Uses synthetic OHLCV data (no price_cache/network needed) since only model
compute time is being measured, not fetch time. Model loads once; each
pred_len is timed after a warmup call to exclude one-off CUDA kernel
compilation from the numbers.

Usage:
    uv run scripts/benchmark_pred_len.py
    uv run scripts/benchmark_pred_len.py --pred-lens 1,2,3,5,10 --repeats 5
"""
import argparse
import statistics as st
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "strategy")
sys.path.insert(0, ".")
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
from kairos.models import resolve as resolve_model  # noqa: E402

N_ASSETS = 3
LOOKBACK = 300
PRED_SAMPLES = 100  # matches PRED_SAMPLES in strategy/kairos_strategies.py


def make_series(lookback: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, lookback))
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.1, lookback),
        "high": close + abs(rng.normal(0, 0.5, lookback)),
        "low": close - abs(rng.normal(0, 0.5, lookback)),
        "close": close,
        "volume": abs(rng.normal(1e6, 1e5, lookback)),
    }, index=idx)
    return df, pd.Series(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-lens", default="1,2,3,5,10")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default="base")
    ap.add_argument("--assets", type=int, default=N_ASSETS)
    ap.add_argument("--samples", type=int, default=PRED_SAMPLES)
    args = ap.parse_args()
    pred_lens = [int(p) for p in args.pred_lens.split(",")]

    cfg = resolve_model(args.model)
    print(f"Loading {cfg['model_id']} ...")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model = Kronos.from_pretrained(cfg["model_id"])
    predictor = KronosPredictor(model, tokenizer, max_context=cfg["max_context"])
    print(f"  device={predictor.device}")

    df, x_ts = make_series(LOOKBACK)
    df_list = [df] * args.assets
    x_ts_list = [x_ts] * args.assets

    def run(pred_len: int) -> float:
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=pred_len, freq="D"))
        y_ts_list = [y_ts] * args.assets
        t0 = time.perf_counter()
        predictor.predict_batch(
            df_list, x_ts_list, y_ts_list, pred_len=pred_len,
            sample_count=args.samples, verbose=False,
        )
        return time.perf_counter() - t0

    # Warmup: excludes first-call CUDA kernel compilation from the measured numbers.
    print("Warmup ...")
    run(max(pred_lens))

    results = {}
    for pl in pred_lens:
        times = [run(pl) for _ in range(args.repeats)]
        results[pl] = times
        print(f"pred_len={pl:>3}  median={st.median(times):.3f}s  "
              f"all={[f'{t:.3f}' for t in times]}")

    print(f"\n{args.assets} assets x {args.samples} samples, lookback={LOOKBACK}, model={args.model}")
    print(f"{'pred_len':>10} {'median s':>10} {'s/bar':>10} {'vs pred_len=1':>14}")
    base = st.median(results[pred_lens[0]])
    base_pl = pred_lens[0]
    for pl in pred_lens:
        med = st.median(results[pl])
        print(f"{pl:>10} {med:>10.3f} {med / pl:>10.3f} {med / base:>13.2f}x")


if __name__ == "__main__":
    main()
