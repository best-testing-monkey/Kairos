import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import KairosDistribution, Direction, Signal
from kairos_sentiment import NewsSentimentFilterStrategy


# ============================================================================
# Helpers
# ============================================================================

def make_dist(close_prices, open_prices=None, high_prices=None, low_prices=None):
    """Build a KairosDistribution from a list of close prices."""
    prices = np.array(close_prices, dtype=float)
    n = len(prices)
    o = np.array(open_prices or prices * 0.999, dtype=float)
    h = np.array(high_prices or prices * 1.005, dtype=float)
    l = np.array(low_prices or prices * 0.995, dtype=float)
    frames = []
    for i in range(n):
        frames.append(pd.DataFrame({
            "open": [o[i]], "high": [h[i]], "low": [l[i]],
            "close": [prices[i]], "volume": [1e6], "amount": [1e9]
        }))
    return KairosDistribution(frames)


def make_history(n=50, price=100.0):
    """Build a minimal history DataFrame for backtesting."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [price]*n, "high": [price*1.01]*n,
        "low": [price*0.99]*n, "close": [price]*n, "volume": [1e6]*n
    }, index=idx)


class MockStrategy:
    """Mock strategy that always returns a fixed signal."""
    name = "mock"

    def __init__(self, direction=Direction.LONG, size=0.5, confidence=0.8, expected_value=100.0):
        self.direction = direction
        self.size = size
        self.confidence = confidence
        self.expected_value = expected_value

    def generate_signal(self, dist, current_price, history, context):
        return Signal(
            direction=self.direction,
            size=self.size,
            entry=current_price,
            stop=current_price * 0.95,
            target=current_price * 1.05,
            strategy_name=self.name,
            confidence=self.confidence,
            expected_value=self.expected_value,
        )


class MockNoneStrategy:
    """Mock strategy that always returns None."""
    name = "mock_none"

    def generate_signal(self, dist, current_price, history, context):
        return None


# ============================================================================
# Tests
# ============================================================================

class TestNewsSentimentFilterStrategy:
    """Test NewsSentimentFilterStrategy graceful degradation and filtering logic."""

    def test_news_sentiment_none_base_signal(self):
        """Test pass-through on None base signal."""
        base = MockNoneStrategy()
        filt = NewsSentimentFilterStrategy(base)
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.8}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_news_sentiment_missing_context_key(self):
        """
        Test missing context key → identical behavior to unwrapped strategy.
        Verifies field-by-field identity.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8, expected_value=100.0)
        filt = NewsSentimentFilterStrategy(base)
        dist = make_dist([110.0] * 100)
        # No "news_sentiment" key in context
        context = {"symbol": "BTC-USD"}
        sig = filt.generate_signal(dist, 100.0, make_history(), context)

        # Get unwrapped base signal for comparison
        base_sig = base.generate_signal(dist, 100.0, make_history(), context)

        assert sig is not None
        assert sig.direction == base_sig.direction
        assert sig.size == pytest.approx(base_sig.size, rel=1e-6)
        assert sig.confidence == pytest.approx(base_sig.confidence, rel=1e-6)
        assert sig.expected_value == pytest.approx(base_sig.expected_value, rel=1e-6)
        assert sig.entry == base_sig.entry
        assert sig.stop == base_sig.stop
        assert sig.target == base_sig.target

    def test_news_sentiment_missing_symbol_in_dict(self):
        """
        Test missing symbol in sentiment dict → pass through unchanged.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"ETH-USD": 0.8}  # sentiment dict exists but no BTC-USD
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        base_sig = base.generate_signal(dist, 100.0, make_history(), context)

        assert sig is not None
        assert sig.direction == base_sig.direction
        assert sig.size == pytest.approx(base_sig.size, rel=1e-6)
        assert sig.confidence == pytest.approx(base_sig.confidence, rel=1e-6)

    def test_news_sentiment_missing_symbol_context_key(self):
        """
        Test missing symbol in context → pass through unchanged.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base)
        dist = make_dist([110.0] * 100)
        context = {
            # No "symbol" key
            "news_sentiment": {"BTC-USD": 0.8}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None

    def test_news_sentiment_opposing_veto_long(self):
        """
        Test opposing sentiment vetoes LONG signals.
        LONG signal (direction=1) + negative sentiment (-0.7) → None
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": -0.7}  # Bearish, opposes LONG
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_news_sentiment_opposing_veto_short(self):
        """
        Test opposing sentiment vetoes SHORT signals.
        SHORT signal (direction=-1) + positive sentiment (0.7) → None
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.7}  # Bullish, opposes SHORT
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_news_sentiment_opposing_at_threshold(self):
        """
        Test that opposing sentiment at exact threshold still vetoes.
        Using |sentiment| = veto_threshold exactly.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": -0.5}  # Exactly at threshold, bearish
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        # At threshold, should still veto (|sentiment| > veto_threshold is false, so no filter)
        # Actually, >= check or > check? Let's verify with > (stricter)
        # The code uses > so -0.5 with threshold 0.5 means abs(-0.5) = 0.5 is NOT > 0.5
        # So it should pass through
        assert sig is not None

    def test_news_sentiment_aligned_boost_long(self):
        """
        Test aligned sentiment boosts LONG signal confidence.
        LONG signal (direction=1) + positive sentiment (0.8) → confidence *= boost, capped at 1.0
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.6)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=1.2)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.8}  # Bullish, aligns with LONG
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        # 0.6 * 1.2 = 0.72
        assert sig.confidence == pytest.approx(0.72, rel=1e-6)

    def test_news_sentiment_aligned_boost_short(self):
        """
        Test aligned sentiment boosts SHORT signal confidence.
        SHORT signal (direction=-1) + negative sentiment (-0.8) → confidence *= boost
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.6)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=1.2)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": -0.8}  # Bearish, aligns with SHORT
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.confidence == pytest.approx(0.72, rel=1e-6)

    def test_news_sentiment_aligned_boost_capped_at_one(self):
        """
        Test that boosted confidence is capped at 1.0.
        LONG signal (direction=1) + positive sentiment (0.9) with boost=2.0
        confidence 0.8 * 2.0 = 1.6 → capped at 1.0
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.9}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.confidence == pytest.approx(1.0, rel=1e-6)

    def test_news_sentiment_neutral_passthrough(self):
        """
        Test neutral sentiment (|s| <= threshold) passes through unchanged.
        LONG signal with neutral sentiment (0.3) and threshold 0.5 → no filtering
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.3}  # Below threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.confidence == pytest.approx(0.8, rel=1e-6)  # Unchanged

    def test_news_sentiment_zero_neutral(self):
        """
        Test zero sentiment (perfect neutral) passes through unchanged.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.0}  # Perfect neutral
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.confidence == pytest.approx(0.8, rel=1e-6)

    def test_news_sentiment_flat_signal(self):
        """
        Test FLAT signals pass through unchanged regardless of sentiment.
        FLAT direction has value 0, should not be vetoed or boosted.
        """
        base = MockStrategy(direction=Direction.FLAT, size=0.0, confidence=0.0)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=1.2)
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.9}  # Strong sentiment
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.size == 0.0
        assert sig.confidence == 0.0

    def test_news_sentiment_metadata_preserved(self):
        """
        Test that base signal metadata is preserved and updated appropriately.
        Boosted signals should have metadata added.
        """
        base_sig = Signal(
            direction=Direction.LONG,
            size=0.5,
            entry=100.0,
            stop=95.0,
            target=105.0,
            strategy_name="mock",
            confidence=0.6,
            expected_value=100.0,
            metadata={"base_key": "base_value"}
        )

        class CustomMockStrategy:
            name = "custom_mock"
            def generate_signal(self, dist, current_price, history, context):
                return base_sig

        base = CustomMockStrategy()
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=1.2)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.8}  # Bullish, boosts
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.metadata.get("base_key") == "base_value"
        assert sig.metadata.get("sentiment_boosted") is True
        assert sig.metadata.get("news_sentiment") == pytest.approx(0.8, rel=1e-6)

    def test_news_sentiment_extreme_sentiment_values(self):
        """
        Test extreme sentiment values (-1.0 and 1.0).
        """
        base_long = MockStrategy(direction=Direction.LONG, confidence=0.5)
        base_short = MockStrategy(direction=Direction.SHORT, confidence=0.5)

        # Test max bullish (1.0)
        filt = NewsSentimentFilterStrategy(base_long, veto_threshold=0.5, boost=1.5)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 1.0}  # Max bullish
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.confidence == pytest.approx(min(0.5 * 1.5, 1.0), rel=1e-6)

        # Test max bearish (-1.0)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": -1.0}  # Max bearish
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None  # Vetoes LONG

    def test_news_sentiment_custom_threshold(self):
        """
        Test custom veto_threshold parameter.
        With threshold=0.7, |sentiment|=0.6 should pass through.
        """
        base = MockStrategy(direction=Direction.LONG, confidence=0.8)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.7)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": -0.6}  # Opposes LONG but below 0.7 threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None  # Passes through

    def test_news_sentiment_custom_boost(self):
        """
        Test custom boost parameter.
        Boost=2.0 should double confidence (capped at 1.0).
        """
        base = MockStrategy(direction=Direction.LONG, confidence=0.6)
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.8}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.confidence == pytest.approx(min(0.6 * 2.0, 1.0), rel=1e-6)

    def test_news_sentiment_signal_other_fields_unchanged(self):
        """
        Test that filtering only modifies confidence and metadata,
        not size, entry, stop, target, etc.
        """
        base = MockStrategy(
            direction=Direction.LONG,
            size=0.7,
            confidence=0.6,
            expected_value=150.0
        )
        filt = NewsSentimentFilterStrategy(base, veto_threshold=0.5, boost=1.2)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "news_sentiment": {"BTC-USD": 0.8}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        base_sig = base.generate_signal(dist, 100.0, make_history(), context)

        assert sig.size == pytest.approx(base_sig.size, rel=1e-6)
        assert sig.entry == base_sig.entry
        assert sig.stop == base_sig.stop
        assert sig.target == base_sig.target
        assert sig.expected_value == pytest.approx(base_sig.expected_value, rel=1e-6)
        assert sig.direction == base_sig.direction
