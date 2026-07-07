import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import numpy as np
import pandas as pd
import pytest

from kairos_backtest import Direction, Signal
from kairos_orchestrator import StrategyRegistry, apply_allocator, KairosOrchestrator


# ============================================================================
# Helpers / stubs
# ============================================================================

def make_signal(direction=Direction.LONG, size=0.1, strategy_name="stub"):
    return Signal(
        direction=direction,
        size=size,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name=strategy_name,
        confidence=0.6,
        expected_value=1.0,
        metadata={},
    )


def make_returns_panel(symbols, n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {sym: rng.normal(0, 0.01, n) for sym in symbols}
    return pd.DataFrame(data, index=idx)


class FixedWeightAllocator:
    """Stub allocator returning a fixed weights dict, ignoring inputs."""
    name = "fixed_weight"

    def __init__(self, weights):
        self._weights = weights

    def allocate(self, signals, returns, dists, context):
        return dict(self._weights)


class RaisingAllocator:
    """Stub allocator that always raises."""
    name = "raising"

    def allocate(self, signals, returns, dists, context):
        raise RuntimeError("boom")


class ZeroWeightAllocator:
    """Stub allocator that zeroes out everything."""
    name = "zero_weight"

    def allocate(self, signals, returns, dists, context):
        return {sym: 0.0 for sym in signals}


# ============================================================================
# StrategyRegistry.register_allocator / get_allocator semantics
# ============================================================================

class TestRegistrySemantics:
    def test_register_and_get(self):
        registry = StrategyRegistry()
        assert registry.get_allocator() is None

        allocator = FixedWeightAllocator({"BTC-USD": 0.5})
        registry.register_allocator(allocator)
        assert registry.get_allocator() is allocator

    def test_register_replaces_previous(self):
        registry = StrategyRegistry()
        first = FixedWeightAllocator({"BTC-USD": 0.5})
        second = FixedWeightAllocator({"BTC-USD": 0.3})
        registry.register_allocator(first)
        registry.register_allocator(second)
        assert registry.get_allocator() is second

    def test_register_none_clears(self):
        registry = StrategyRegistry()
        registry.register_allocator(FixedWeightAllocator({}))
        registry.register_allocator(None)
        assert registry.get_allocator() is None


# ============================================================================
# apply_allocator pure function
# ============================================================================

class TestApplyAllocator:
    def test_no_allocator_returns_signals_unchanged(self):
        signals = {"BTC-USD": make_signal(size=0.2), "ETH-USD": make_signal(size=0.3)}
        result = apply_allocator(signals, None, make_returns_panel(["BTC-USD", "ETH-USD"]), {}, {})
        assert result is signals

    def test_single_signal_skips_allocation(self):
        signals = {"BTC-USD": make_signal(size=0.2)}
        allocator = FixedWeightAllocator({"BTC-USD": 0.01})
        result = apply_allocator(signals, allocator, make_returns_panel(["BTC-USD"]), {}, {})
        # <= 1 signal: allocator not invoked, sizes untouched
        assert result["BTC-USD"].size == 0.2

    def test_size_replaced_with_min_of_original_and_abs_weight(self):
        signals = {
            "BTC-USD": make_signal(size=0.5),
            "ETH-USD": make_signal(size=0.05),
        }
        allocator = FixedWeightAllocator({"BTC-USD": 0.2, "ETH-USD": 0.4})
        returns = make_returns_panel(["BTC-USD", "ETH-USD"])
        result = apply_allocator(signals, allocator, returns, {}, {})

        # BTC: original 0.5, weight 0.2 -> capped to 0.2
        assert result["BTC-USD"].size == pytest.approx(0.2)
        # ETH: original 0.05, weight 0.4 -> stays at original (per-signal cap)
        assert result["ETH-USD"].size == pytest.approx(0.05)

    def test_negative_weight_uses_absolute_value(self):
        signals = {
            "BTC-USD": make_signal(size=0.5),
            "ETH-USD": make_signal(size=0.5),
        }
        allocator = FixedWeightAllocator({"BTC-USD": -0.1, "ETH-USD": 0.5})
        returns = make_returns_panel(["BTC-USD", "ETH-USD"])
        result = apply_allocator(signals, allocator, returns, {}, {})
        assert result["BTC-USD"].size == pytest.approx(0.1)
        assert result["ETH-USD"].size == pytest.approx(0.5)

    def test_zero_weight_drops_signal(self):
        signals = {
            "BTC-USD": make_signal(size=0.5),
            "ETH-USD": make_signal(size=0.5),
        }
        allocator = ZeroWeightAllocator()
        returns = make_returns_panel(["BTC-USD", "ETH-USD"])
        result = apply_allocator(signals, allocator, returns, {}, {})
        assert result == {}

    def test_allocator_exception_preserves_original_sizes(self):
        signals = {
            "BTC-USD": make_signal(size=0.5),
            "ETH-USD": make_signal(size=0.3),
        }
        allocator = RaisingAllocator()
        returns = make_returns_panel(["BTC-USD", "ETH-USD"])
        result = apply_allocator(signals, allocator, returns, {}, {})
        assert result["BTC-USD"].size == 0.5
        assert result["ETH-USD"].size == 0.3

    def test_symbol_missing_from_weights_kept_unchanged(self):
        signals = {
            "BTC-USD": make_signal(size=0.5),
            "ETH-USD": make_signal(size=0.3),
        }
        # Allocator only opines on BTC-USD
        allocator = FixedWeightAllocator({"BTC-USD": 0.1})
        returns = make_returns_panel(["BTC-USD", "ETH-USD"])
        result = apply_allocator(signals, allocator, returns, {}, {})
        assert result["BTC-USD"].size == pytest.approx(0.1)
        assert result["ETH-USD"].size == 0.3


# ============================================================================
# Context enrichment: returns_window / realized_vol
# ============================================================================

def make_history(n=50, price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes, "volume": [1e6] * n,
    }, index=idx)


def _make_dummy_orchestrator():
    def dummy_predict(signal, **kwargs):
        return []
    return KairosOrchestrator(predict_fn=dummy_predict, assets=["BTC-USD"])


class TestContextEnrichment:
    def test_returns_window_shape_and_values(self):
        orch = _make_dummy_orchestrator()
        histories = {
            "BTC-USD": make_history(seed=1),
            "ETH-USD": make_history(seed=2),
        }
        returns_window = orch._compute_returns_window(histories)
        assert list(returns_window.columns) == ["BTC-USD", "ETH-USD"]
        assert len(returns_window) > 0
        # log returns should not be identical to raw prices
        assert not returns_window.isnull().all().all()

    def test_realized_vol_per_symbol(self):
        orch = _make_dummy_orchestrator()
        histories = {
            "BTC-USD": make_history(seed=1),
            "ETH-USD": make_history(seed=2),
        }
        returns_window = orch._compute_returns_window(histories)
        realized_vol = orch._compute_realized_vol(returns_window)
        assert set(realized_vol.keys()) == {"BTC-USD", "ETH-USD"}
        for v in realized_vol.values():
            assert v >= 0

    def test_empty_histories_returns_empty(self):
        orch = _make_dummy_orchestrator()
        assert orch._compute_returns_window({}).empty
        assert orch._compute_realized_vol(pd.DataFrame()) == {}
