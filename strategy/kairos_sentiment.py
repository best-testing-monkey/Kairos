"""
News sentiment filter strategies.

CONTEXT-KEY CONTRACT & GRACEFUL DEGRADATION:
This module implements filter wrappers that read external sentiment signals
from the context dict and adjust trading signals accordingly. Each filter
follows the graceful degradation pattern: if the required context key is
absent or the symbol is not found, the filter returns the base strategy's
signal UNCHANGED. This allows sentiment strategies to be registered before
any data feed exists, enabling safe pipeline composition.

Key contracts:
  - context["news_sentiment"][symbol] ∈ [-1, 1] (float)
    Positive values indicate bullish sentiment, negative values bearish.
    Missing keys or symbols → no filtering applied.

The NewsSentimentFilterStrategy wrapper:
  - Wraps a base strategy
  - Reads news sentiment for the current symbol
  - Vetoes signals fighting strong opposing sentiment (|s| > veto_threshold)
  - Boosts confidence when aligned (|s| > veto_threshold, same sign)
  - Returns Signal or None, never dict
"""

from kairos_backtest import Strategy, Signal, Direction


class NewsSentimentFilterStrategy(Strategy):
    """
    Filters signals based on news sentiment context.

    Wraps a base strategy and applies news sentiment validation:
      - Vetoes signals that fight strong opposing sentiment (|sentiment| > veto_threshold
        and opposite sign to signal direction).
      - Boosts confidence when sentiment aligns with signal direction.
      - Gracefully degrades: if context["news_sentiment"][symbol] is missing,
        returns base signal unchanged.

    Args:
        base_strategy: The wrapped strategy to filter.
        veto_threshold: Minimum |sentiment| magnitude to trigger veto or boost.
                        Default 0.5 (require moderate confidence to override).
        boost: Confidence multiplier when sentiment aligns with signal.
               Default 1.2 (20% confidence boost). Final confidence capped at 1.0.
    """
    name = "news_sentiment_filter"

    def __init__(self, base_strategy: Strategy, veto_threshold: float = 0.5,
                 boost: float = 1.2):
        self.base_strategy = base_strategy
        self.veto_threshold = veto_threshold
        self.boost = boost

    def generate_signal(self, dist, current_price, history, context):
        # Get base signal from wrapped strategy
        signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if signal is None:
            return None

        # Graceful degradation: if sentiment context missing, return base signal unchanged
        sentiment_dict = context.get("news_sentiment")
        if sentiment_dict is None:
            return signal

        symbol = context.get("symbol")
        if symbol is None or symbol not in sentiment_dict:
            return signal

        sentiment = sentiment_dict[symbol]

        # Only apply filtering if sentiment magnitude is above threshold
        if abs(sentiment) <= self.veto_threshold:
            return signal

        # Determine sign of signal direction: LONG=1, SHORT=-1, FLAT=0
        direction_sign = signal.direction.value

        # If FLAT signal or direction sign is 0, pass through unchanged
        if direction_sign == 0:
            return signal

        # Determine sentiment sign: positive (1) or negative (-1)
        sentiment_sign = 1 if sentiment > 0 else -1

        # Veto: signal direction opposes strong sentiment
        if direction_sign != sentiment_sign:
            return None

        # Boost: signal direction aligns with strong sentiment
        if direction_sign == sentiment_sign:
            signal.confidence = min(signal.confidence * self.boost, 1.0)
            if not hasattr(signal, 'metadata'):
                signal.metadata = {}
            signal.metadata["sentiment_boosted"] = True
            signal.metadata["news_sentiment"] = sentiment

        return signal
