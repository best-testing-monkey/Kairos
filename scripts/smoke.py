"""Fast smoke test for the Kairos orchestrator, no model/GPU/network needed.

Encapsulates the ad-hoc heredoc that used to get regenerated in nearly every
session: builds a dummy predict_fn, constructs a KairosOrchestrator on a
synthetic asset, and runs a tiny backtest to confirm the strategy stack wires
up end to end.

Run with:
    uv run --with pytest python scripts/smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))

import numpy as np
import pandas as pd

from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig


def make_history(n=260, price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6,
    }, index=idx)


def dummy_predict(signal: pd.DataFrame, **kwargs):
    """Fake predict_fn: returns a handful of frames scattered around the last close."""
    last_close = float(signal["close"].iloc[-1])
    frames = []
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = last_close * (1 + rng.normal(0, 0.01))
        frames.append(pd.DataFrame({
            "open": [c], "high": [c * 1.01], "low": [c * 0.99],
            "close": [c], "volume": [1e6], "amount": [1e8],
        }))
    return frames


def main():
    config = OrchestratorConfig(verbose=True)
    orch = KairosOrchestrator(predict_fn=dummy_predict, assets=["BTC-USD"], config=config)
    histories = {"BTC-USD": make_history()}
    results = orch.run_backtest(histories, lookback=200)
    print("Smoke test OK. Result keys:", list(results.keys()))


if __name__ == "__main__":
    main()
