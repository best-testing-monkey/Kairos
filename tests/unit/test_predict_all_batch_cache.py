"""Tests for the predict_all_batch <-> kairos_predcache wiring (Feature 2)."""

import pandas as pd
import pytest

import kairos_predcache as pc
import kairos_strategies as ks
from kairos_backtest import KairosSettings


def _make_asset_df(n=310, base=100.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [base] * n, "high": [base + 1] * n, "low": [base - 1] * n,
        "close": [base + i * 0.01 for i in range(n)],
        "volume": [1000.0] * n,
    }, index=dates)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    # Reset module-level in-process caches so each test starts clean, and
    # give KairosSettings deterministic values.
    ks._prediction_cache.clear()
    ks._dist_cache.clear()
    monkeypatch.setattr(KairosSettings, "lookback", 300)
    monkeypatch.setattr(KairosSettings, "pred_samples", 5)
    monkeypatch.setattr(KairosSettings, "interval", "1d")
    monkeypatch.setattr(KairosSettings, "model", None)
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    pc._singleton = None
    pc._singleton_dir = None
    yield
    ks._prediction_cache.clear()
    ks._dist_cache.clear()
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    pc._singleton = None
    pc._singleton_dir = None


class _StubPredictor:
    """Counts how many symbols the model was actually invoked for."""

    def __init__(self):
        self.calls = 0
        self.symbols_predicted = 0

    def predict_batch(self, df_list, x_ts_list, y_ts_list, pred_len, sample_count,
                       return_samples, verbose):
        self.calls += 1
        self.symbols_predicted += len(df_list)
        out = []
        for i in range(len(df_list)):
            samples = []
            for s in range(sample_count):
                samples.append(pd.DataFrame({
                    "open": [100.0 + s], "high": [101.0 + s], "low": [99.0 + s],
                    "close": [100.0 + s * 0.1], "volume": [1000.0], "amount": [100000.0],
                }, index=[x_ts_list[i].iloc[-1]]))
            out.append(samples)
        return out


def _patch_common(monkeypatch, stub_predictor):
    monkeypatch.setattr(ks, "bt_predictor", stub_predictor)
    monkeypatch.setattr(ks, "_ensure_model_loaded", lambda *a, **kw: None)

    def fake_to_kronos_frame(df, lookback, amount="auto"):
        x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]].copy()
        x_ts = pd.Series(x_df.index)
        return x_df, x_ts

    def fake_future_timestamps(last_ts, interval, n, calendar, tz):
        return pd.Series([last_ts + pd.Timedelta(days=1)])

    monkeypatch.setattr(ks, "to_kronos_frame", fake_to_kronos_frame)
    monkeypatch.setattr(ks, "future_timestamps", fake_future_timestamps)


def test_cache_disabled_by_default_calls_model_every_time(monkeypatch):
    stub = _StubPredictor()
    _patch_common(monkeypatch, stub)

    assets = {"BTC-USD": _make_asset_df()}
    ks.predict_all_batch(assets)
    assert stub.calls == 1
    assert stub.symbols_predicted == 1

    # Simulate a fresh subprocess: clear the in-process cache. Without
    # KAIROS_PRED_CACHE_DIR set, the model must be called again.
    ks._prediction_cache.clear()
    ks._dist_cache.clear()
    ks.predict_all_batch(assets)
    assert stub.calls == 2
    assert stub.symbols_predicted == 2


def test_shared_cache_hit_across_simulated_subprocess_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    stub = _StubPredictor()
    _patch_common(monkeypatch, stub)

    assets = {"BTC-USD": _make_asset_df()}
    result1 = ks.predict_all_batch(assets)
    assert stub.calls == 1
    assert stub.symbols_predicted == 1

    # Simulate a new subprocess: clear the in-process caches, but the shared
    # disk cache (KAIROS_PRED_CACHE_DIR) persists.
    ks._prediction_cache.clear()
    ks._dist_cache.clear()

    result2 = ks.predict_all_batch(assets)
    # No new model calls: the prediction was reused from the shared cache.
    assert stub.calls == 1
    assert stub.symbols_predicted == 1
    assert result2["BTC-USD"].current_price == result1["BTC-USD"].current_price


def test_mixed_hit_and_miss_batch_only_predicts_misses(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    stub = _StubPredictor()
    _patch_common(monkeypatch, stub)

    assets1 = {"BTC-USD": _make_asset_df(base=100.0)}
    ks.predict_all_batch(assets1)
    assert stub.symbols_predicted == 1

    ks._prediction_cache.clear()
    ks._dist_cache.clear()

    # BTC-USD is an exact repeat (cache hit); ETH-USD is new (cache miss).
    assets2 = {
        "BTC-USD": _make_asset_df(base=100.0),
        "ETH-USD": _make_asset_df(base=200.0),
    }
    result = ks.predict_all_batch(assets2)
    # Only the miss (ETH-USD) should have gone through the model.
    assert stub.symbols_predicted == 2  # 1 (first call) + 1 (ETH-USD miss)
    assert set(result.keys()) == {"BTC-USD", "ETH-USD"}


def test_different_content_hash_is_a_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    stub = _StubPredictor()
    _patch_common(monkeypatch, stub)

    assets_a = {"BTC-USD": _make_asset_df(base=100.0)}
    ks.predict_all_batch(assets_a)
    assert stub.symbols_predicted == 1

    ks._prediction_cache.clear()
    ks._dist_cache.clear()

    # Same symbol, same last-bar timestamp, but different lookback content
    # (different base price) -> content_hash differs -> must be a miss.
    assets_b = {"BTC-USD": _make_asset_df(base=999.0)}
    ks.predict_all_batch(assets_b)
    assert stub.symbols_predicted == 2
