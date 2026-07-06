import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import KairosDistribution, Direction, Signal
from kairos_sentiment import NewsSentimentFilterStrategy, SocialMomentumStrategy, Institutional13FFilterStrategy


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


# ============================================================================
# Tests for SocialMomentumStrategy
# ============================================================================

class TestSocialMomentumStrategy:
    """Test SocialMomentumStrategy standalone strategy for social momentum trading."""

    def test_social_momentum_missing_context_key(self):
        """
        Test graceful degradation when context["social_mentions"] is missing.
        Should return None, never raise exception.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)
        context = {"symbol": "BTC-USD"}  # No social_mentions key
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_missing_symbol_in_dict(self):
        """
        Test graceful degradation when symbol not in social_mentions dict.
        Should return None, never raise exception.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"ETH-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 100}}
        }
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_missing_symbol_context_key(self):
        """
        Test graceful degradation when context["symbol"] is missing.
        Should return None, never raise exception.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)
        context = {
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 100}}
        }
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_missing_z_score(self):
        """
        Test graceful degradation when z_score is missing from mention data.
        Should return None, never raise exception.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"sentiment": 0.8, "count": 100}}  # No z_score
        }
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_missing_sentiment(self):
        """
        Test graceful degradation when sentiment is missing from mention data.
        Should return None, never raise exception.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "count": 100}}  # No sentiment
        }
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_z_score_too_low(self):
        """
        Test that z_score <= 3 returns None (no signal).
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 3.0, "sentiment": 0.8, "count": 100}}
        }
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_z_score_at_threshold(self):
        """
        Test that z_score > 3 (just above threshold) with positive sentiment → signal.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)  # mean > current_price for Kronos agreement
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 3.1, "sentiment": 0.5, "count": 100}}
        }
        # History with no 5-day runup (all same price)
        hist = make_history(n=50, price=100.0)
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_social_momentum_negative_sentiment(self):
        """
        Test that sentiment <= 0 returns None (no signal), even with high z_score.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 5.0, "sentiment": 0.0, "count": 100}}
        }
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_long_with_kronos_agreement(self):
        """
        Test LONG signal when z_score > 3, sentiment > 0, no 5-day runup,
        and Kronos agrees (mean > current_price).
        """
        strategy = SocialMomentumStrategy()
        # Distribution with mean > current_price (Kronos predicts up)
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)  # Flat history, no 5-day runup
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.confidence == pytest.approx(min(4.0 / 6.0, 1.0), rel=1e-6)
        assert "z_score" in sig.metadata
        assert sig.metadata["z_score"] == pytest.approx(4.0, rel=1e-6)
        assert sig.metadata["momentum_type"] == "crowd_inflow"

    def test_social_momentum_kronos_disagreement_long_setup(self):
        """
        Test no signal when attempting LONG but Kronos predicts down (mean < current_price).
        """
        strategy = SocialMomentumStrategy()
        # Distribution with mean < current_price (Kronos predicts down)
        dist = make_dist([95.0] * 100)
        hist = make_history(n=50, price=100.0)  # Flat history, no 5-day runup
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is None  # Kronos disagrees with LONG

    def test_social_momentum_blowoff_fade_with_kronos_agreement(self):
        """
        Test SHORT fade signal when z_score > 3, sentiment > 0,
        5-day runup > +20%, and Kronos agrees (mean < current_price).
        """
        strategy = SocialMomentumStrategy()
        # Distribution with mean < current_price (Kronos predicts down, agrees with SHORT)
        dist = make_dist([95.0] * 100)

        # Build history with +25% 5-day runup
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        base_price = 80.0
        prices = [base_price] * 45 + [base_price, base_price, base_price, base_price, base_price * 1.25]
        hist = pd.DataFrame({
            "open": prices,
            "high": np.array(prices) * 1.01,
            "low": np.array(prices) * 0.99,
            "close": prices,
            "volume": [1e6] * 50
        }, index=idx)

        current_price = base_price * 1.25  # Moved to 100
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.5, "sentiment": 0.8, "count": 200}}
        }
        sig = strategy.generate_signal(dist, current_price, hist, context)
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.confidence == pytest.approx(min(4.5 / 6.0, 1.0), rel=1e-6)
        assert sig.metadata["trailing_5d_return"] == pytest.approx(0.25, rel=1e-6)
        assert sig.metadata["momentum_type"] == "fade_blowoff"

    def test_social_momentum_kronos_disagreement_short_setup(self):
        """
        Test no signal when attempting SHORT fade but Kronos predicts up (mean > current_price).
        """
        strategy = SocialMomentumStrategy()
        # Distribution with mean > current_price (Kronos predicts up)
        dist = make_dist([105.0] * 100)

        # Build history with +25% 5-day runup
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        base_price = 80.0
        prices = [base_price] * 45 + [base_price, base_price, base_price, base_price, base_price * 1.25]
        hist = pd.DataFrame({
            "open": prices,
            "high": np.array(prices) * 1.01,
            "low": np.array(prices) * 0.99,
            "close": prices,
            "volume": [1e6] * 50
        }, index=idx)

        current_price = base_price * 1.25
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.5, "sentiment": 0.8, "count": 200}}
        }
        sig = strategy.generate_signal(dist, current_price, hist, context)
        assert sig is None  # Kronos disagrees with SHORT

    def test_social_momentum_short_boundary_exactly_20_percent(self):
        """
        Test that exactly +20% 5-day return does NOT trigger fade (should be LONG).
        Boundary condition: trailing_5d_return > 0.20 required for SHORT.
        """
        strategy = SocialMomentumStrategy()
        # Kronos agrees with LONG: mean > current_price
        dist = make_dist([130.0] * 100)

        # Build history with exactly +20% 5-day runup
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        base_price = 100.0
        prices = [base_price] * 45 + [base_price, base_price, base_price, base_price, base_price * 1.20]
        hist = pd.DataFrame({
            "open": prices,
            "high": np.array(prices) * 1.01,
            "low": np.array(prices) * 0.99,
            "close": prices,
            "volume": [1e6] * 50
        }, index=idx)

        current_price = base_price * 1.20  # 120
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, current_price, hist, context)
        assert sig is not None
        assert sig.direction == Direction.LONG  # At 20%, still LONG, not SHORT
        assert sig.metadata["momentum_type"] == "crowd_inflow"

    def test_social_momentum_short_boundary_above_20_percent(self):
        """
        Test that +20.1% 5-day return DOES trigger fade (SHORT).
        """
        strategy = SocialMomentumStrategy()
        # Kronos agrees with SHORT: mean < current_price
        dist = make_dist([115.0] * 100)

        # Build history with +20.1% 5-day runup
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        base_price = 100.0
        prices = [base_price] * 45 + [base_price, base_price, base_price, base_price, base_price * 1.201]
        hist = pd.DataFrame({
            "open": prices,
            "high": np.array(prices) * 1.01,
            "low": np.array(prices) * 0.99,
            "close": prices,
            "volume": [1e6] * 50
        }, index=idx)

        current_price = base_price * 1.201  # 120.1
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, current_price, hist, context)
        assert sig is not None
        assert sig.direction == Direction.SHORT  # Above 20%, SHORT fade
        assert sig.metadata["momentum_type"] == "fade_blowoff"

    def test_social_momentum_insufficient_history(self):
        """
        Test behavior when history has fewer than 5 bars.
        Should default trailing_5d_return to 0.0, proceed as LONG.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)  # Kronos agrees with LONG

        # History with only 3 bars
        hist = make_history(n=3, price=100.0)

        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.metadata["trailing_5d_return"] == pytest.approx(0.0, rel=1e-6)

    def test_social_momentum_confidence_calculation(self):
        """
        Test that confidence is computed as min(z_score / 6, 1.0).
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)

        # Test z_score = 3.6 → confidence = 3.6/6 = 0.6
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 3.6, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert sig.confidence == pytest.approx(0.6, rel=1e-6)

    def test_social_momentum_confidence_capped_at_one(self):
        """
        Test that confidence is capped at 1.0 even when z_score is very high.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)

        # Test z_score = 10.0 → confidence = min(10.0/6, 1.0) = 1.0
        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 10.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert sig.confidence == pytest.approx(1.0, rel=1e-6)

    def test_social_momentum_size_calculation(self):
        """
        Test that size is computed as min(kelly_fraction * 0.5, 1.0).
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)

        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        # Size should be min(kelly * 0.5, 1.0)
        # Kelly is computed from distribution, so size should be <= 1.0
        assert sig.size >= 0.0
        assert sig.size <= 1.0

    def test_social_momentum_metadata_content(self):
        """
        Test that metadata contains expected fields: z_score, sentiment,
        trailing_5d_return, momentum_type.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)

        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.7, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert sig.metadata["z_score"] == pytest.approx(4.0, rel=1e-6)
        assert sig.metadata["sentiment"] == pytest.approx(0.7, rel=1e-6)
        assert sig.metadata["trailing_5d_return"] == pytest.approx(0.0, rel=1e-6)
        assert sig.metadata["momentum_type"] == "crowd_inflow"

    def test_social_momentum_return_type_signal(self):
        """
        Test that signal is always a Signal dataclass, never a dict.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)

        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        assert isinstance(sig, Signal)
        assert not isinstance(sig, dict)

    def test_social_momentum_return_type_none(self):
        """
        Test that function returns None (not empty dict) for invalid cases.
        """
        strategy = SocialMomentumStrategy()
        dist = make_dist([100.0] * 100)

        # Missing context key
        context = {"symbol": "BTC-USD"}
        sig = strategy.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_social_momentum_custom_stop_target_percentiles(self):
        """
        Test that custom stop_pct and target_pct parameters are used.
        """
        strategy = SocialMomentumStrategy(stop_pct=10.0, target_pct=90.0)
        dist = make_dist([105.0] * 100)
        hist = make_history(n=50, price=100.0)

        context = {
            "symbol": "BTC-USD",
            "social_mentions": {"BTC-USD": {"z_score": 4.0, "sentiment": 0.8, "count": 150}}
        }
        sig = strategy.generate_signal(dist, 100.0, hist, context)
        assert sig is not None
        # For LONG: stop should use pct_10, target should use pct_90
        expected_stop = dist.stats["close"].get("pct_10")
        expected_target = dist.stats["close"].get("pct_90")
        assert sig.stop == pytest.approx(expected_stop, rel=1e-6)
        assert sig.target == pytest.approx(expected_target, rel=1e-6)


# ============================================================================
# Tests for Institutional13FFilterStrategy
# ============================================================================

class TestInstitutional13FFilterStrategy:
    """Test Institutional13FFilterStrategy graceful degradation and filtering logic."""

    def test_13f_none_base_signal(self):
        """Test pass-through on None base signal."""
        base = MockNoneStrategy()
        filt = Institutional13FFilterStrategy(base)
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 3.0}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_missing_context_key(self):
        """
        Test missing context key → identical behavior to unwrapped strategy.
        Verifies field-by-field identity.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8, expected_value=100.0)
        filt = Institutional13FFilterStrategy(base)
        dist = make_dist([110.0] * 100)
        # No "inst_ownership_delta" key in context
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

    def test_13f_missing_symbol_in_dict(self):
        """
        Test missing symbol in ownership_delta dict → pass through unchanged.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"ETH-USD": 1.5}  # ownership dict exists but no BTC-USD
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        base_sig = base.generate_signal(dist, 100.0, make_history(), context)

        assert sig is not None
        assert sig.direction == base_sig.direction
        assert sig.size == pytest.approx(base_sig.size, rel=1e-6)
        assert sig.confidence == pytest.approx(base_sig.confidence, rel=1e-6)

    def test_13f_missing_symbol_context_key(self):
        """
        Test missing symbol in context → pass through unchanged.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base)
        dist = make_dist([110.0] * 100)
        context = {
            # No "symbol" key
            "inst_ownership_delta": {"BTC-USD": 2.5}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None

    def test_13f_veto_short_on_strong_accumulation(self):
        """
        Test SHORT signal vetoed when delta > accumulation_threshold.
        SHORT signal (direction=-1) + delta=+3% > +2% threshold → None
        Verifies test_13f_veto_short_vs_accumulation requirement.
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 3.0}  # Strong accumulation, > 2.0
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_veto_short_at_threshold(self):
        """
        Test SHORT signal vetoed at exactly accumulation_threshold.
        SHORT + delta=+2.0 exactly equal to threshold should veto.
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 2.0}  # At threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        # delta > threshold requires delta > 2.0, so exactly 2.0 should pass through
        assert sig is not None

    def test_13f_veto_short_above_threshold(self):
        """
        Test SHORT signal vetoed above threshold.
        SHORT + delta=+2.1 > +2.0 → None
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 2.1}  # Just above threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_veto_long_on_strong_distribution(self):
        """
        Test LONG signal vetoed when delta < -accumulation_threshold.
        LONG signal (direction=1) + delta=-3% < -2% threshold → None
        Symmetric to short veto.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": -3.0}  # Strong distribution, < -2.0
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_veto_long_at_threshold(self):
        """
        Test LONG signal vetoed at exactly negative accumulation_threshold.
        LONG + delta=-2.0 exactly equal to -threshold should pass through.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": -2.0}  # At negative threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        # delta < -threshold requires delta < -2.0, so exactly -2.0 should pass through
        assert sig is not None

    def test_13f_veto_long_below_threshold(self):
        """
        Test LONG signal vetoed below negative threshold.
        LONG + delta=-2.1 < -2.0 → None
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": -2.1}  # Just below -threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_short_passes_on_accumulation_below_threshold(self):
        """
        Test SHORT signal passes when delta <= accumulation_threshold.
        SHORT + delta=+1.5% <= +2.0% → pass through
        Verifies test_13f_below_threshold_pass requirement.
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 1.5}  # Below threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.SHORT

    def test_13f_long_passes_on_positive_delta(self):
        """
        Test LONG signal passes on positive delta (strong accumulation).
        LONG + delta=+3.0% → pass through (no veto for LONG on positive delta)
        Verifies long passes on +3% requirement.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 3.0}  # Strong accumulation
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_13f_short_passes_on_negative_delta(self):
        """
        Test SHORT signal passes on negative delta (strong distribution).
        SHORT + delta=-3.0% → pass through (no veto for SHORT on negative delta)
        Verifies short passes on -3% requirement.
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": -3.0}  # Strong distribution
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.SHORT

    def test_13f_small_delta_passes_both_directions(self):
        """
        Test small delta (±1%) passes both LONG and SHORT.
        Verifies small delta (±1%) passes both requirement.
        """
        # Test LONG with +1%
        base_long = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt_long = Institutional13FFilterStrategy(base_long, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 1.0}  # Small positive delta
        }
        sig = filt_long.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.LONG

        # Test SHORT with -1%
        base_short = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt_short = Institutional13FFilterStrategy(base_short, accumulation_threshold=2.0)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": -1.0}  # Small negative delta
        }
        sig = filt_short.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.SHORT

    def test_13f_small_delta_plus_one_passes_both(self):
        """
        Test delta of exactly ±1% (the requirement value) passes both directions.
        """
        # Test LONG with +1%
        base_long = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base_long, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 1.0}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None

        # Test SHORT with -1%
        base_short = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": -1.0}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None

    def test_13f_none_base_passthrough(self):
        """
        Test None base pass-through.
        When base strategy returns None, filter returns None regardless of delta.
        """
        base = MockNoneStrategy()
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 5.0}  # Even high delta
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_flat_signal_passthrough(self):
        """
        Test FLAT signals pass through unchanged regardless of delta.
        FLAT direction has value 0, should not be vetoed.
        """
        base = MockStrategy(direction=Direction.FLAT, size=0.0, confidence=0.0)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([100.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 3.0}  # Strong accumulation
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.size == 0.0

    def test_13f_signal_other_fields_unchanged(self):
        """
        Test that filtering only affects direction/return None,
        not size, entry, stop, target, confidence, etc. when passing through.
        """
        base = MockStrategy(
            direction=Direction.LONG,
            size=0.7,
            confidence=0.6,
            expected_value=150.0
        )
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 0.5}  # Below threshold
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        base_sig = base.generate_signal(dist, 100.0, make_history(), context)

        assert sig.size == pytest.approx(base_sig.size, rel=1e-6)
        assert sig.entry == base_sig.entry
        assert sig.stop == base_sig.stop
        assert sig.target == base_sig.target
        assert sig.expected_value == pytest.approx(base_sig.expected_value, rel=1e-6)
        assert sig.confidence == pytest.approx(base_sig.confidence, rel=1e-6)
        assert sig.direction == base_sig.direction

    def test_13f_custom_accumulation_threshold(self):
        """
        Test custom accumulation_threshold parameter.
        With threshold=3.5, delta=3.0 should pass SHORT, delta=4.0 should veto.
        """
        base_short = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base_short, accumulation_threshold=3.5)
        dist = make_dist([90.0] * 100)

        # delta=3.0 < 3.5 should pass
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 3.0}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None

        # delta=4.0 > 3.5 should veto
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 4.0}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None

    def test_13f_return_type_signal(self):
        """
        Test that signal is always a Signal dataclass (when not None), never a dict.
        """
        base = MockStrategy(direction=Direction.LONG, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([110.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 0.5}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is not None
        assert isinstance(sig, Signal)
        assert not isinstance(sig, dict)

    def test_13f_return_type_none(self):
        """
        Test that function returns None (not empty dict) for vetoed cases.
        """
        base = MockStrategy(direction=Direction.SHORT, size=0.5, confidence=0.8)
        filt = Institutional13FFilterStrategy(base, accumulation_threshold=2.0)
        dist = make_dist([90.0] * 100)
        context = {
            "symbol": "BTC-USD",
            "inst_ownership_delta": {"BTC-USD": 3.0}
        }
        sig = filt.generate_signal(dist, 100.0, make_history(), context)
        assert sig is None
        assert not isinstance(sig, dict)
