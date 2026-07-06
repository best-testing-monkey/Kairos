"""
Sentiment filter and social momentum strategies.

CONTEXT-KEY CONTRACT & GRACEFUL DEGRADATION:
This module implements filter wrappers and standalone strategies that read
external sentiment signals from the context dict. Each follows the graceful
degradation pattern: if the required context key is absent or the symbol is
not found, the filter returns the base strategy's signal UNCHANGED, or the
standalone strategy returns None. This allows sentiment strategies to be
registered before any data feed exists, enabling safe pipeline composition.

Key contracts:
  - context["news_sentiment"][symbol] ∈ [-1, 1] (float)
    Positive values indicate bullish sentiment, negative values bearish.
    Missing keys or symbols → no filtering applied.

  - context["social_mentions"][symbol] = {count, z_score, sentiment} (dict)
    count: int, z_score: float, sentiment: float ∈ [-1, 1]
    Missing keys or symbols → standalone strategy returns None.

  - context["econ_events"] = list of {date, impact} (list of dicts)
    date: str (YYYY-MM-DD), date, or datetime object
    impact: "high", "medium", "low" (str)
    Missing context key or current_date → pass-through unchanged.

The NewsSentimentFilterStrategy wrapper:
  - Wraps a base strategy
  - Reads news sentiment for the current symbol
  - Vetoes signals fighting strong opposing sentiment (|s| > veto_threshold)
  - Boosts confidence when aligned (|s| > veto_threshold, same sign)
  - Returns Signal or None, never dict

The SocialMomentumStrategy standalone strategy:
  - Reads social_mentions: z_score and sentiment
  - Momentum LONG when z_score > 3 and sentiment > 0 (crowd inflow)
  - FADE (SHORT) when z_score > 3, sentiment > 0, but 5-day price return > +20%
  - Gated on Kronos direction agreement
  - Gracefully degrades: missing context/symbol → None
  - Returns Signal or None, never dict

The EconCalendarGuardStrategy wrapper:
  - Wraps a base strategy
  - Reads context["econ_events"] (list of {date, impact})
  - Reads context["current_date"]
  - Vetoes new entries if HIGH-impact events fall within (current_date, current_date + guard_days]
  - Exposes should_tighten_stops(context) method for orchestrator to consult
  - Handles string and date/datetime objects for dates
  - Gracefully degrades: missing context keys → pass-through unchanged
  - Returns Signal or None, never dict
"""

import numpy as np
from datetime import datetime, date, timedelta
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


class Institutional13FFilterStrategy(Strategy):
    """
    Filters signals based on institutional 13F ownership changes.

    Wraps a base strategy and applies 13F institutional ownership filtering:
      - Vetoes SHORT signals when institutional ownership delta > accumulation_threshold
        (strong accumulation indicates institutional support, risky to short).
      - Vetoes LONG signals when institutional ownership delta < -accumulation_threshold
        (strong distribution indicates institutional caution, risky to go long).
      - Gracefully degrades: if context["inst_ownership_delta"][symbol] is missing,
        returns base signal unchanged.

    Args:
        base_strategy: The wrapped strategy to filter.
        accumulation_threshold: Quarterly ownership change in % points to trigger veto.
                               Default 2.0 (veto short on > +2%, veto long on < -2%).
    """
    name = "institutional_13f_filter"

    def __init__(self, base_strategy: Strategy, accumulation_threshold: float = 2.0):
        self.base_strategy = base_strategy
        self.accumulation_threshold = accumulation_threshold

    def generate_signal(self, dist, current_price, history, context):
        # Get base signal from wrapped strategy
        signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if signal is None:
            return None

        # Graceful degradation: if 13F context missing, return base signal unchanged
        ownership_delta_dict = context.get("inst_ownership_delta")
        if ownership_delta_dict is None:
            return signal

        symbol = context.get("symbol")
        if symbol is None or symbol not in ownership_delta_dict:
            return signal

        delta = ownership_delta_dict[symbol]

        # Determine sign of signal direction: LONG=1, SHORT=-1, FLAT=0
        direction_sign = signal.direction.value

        # If FLAT signal or direction sign is 0, pass through unchanged
        if direction_sign == 0:
            return signal

        # Veto SHORT when delta > accumulation_threshold (strong institutional accumulation)
        if direction_sign == -1 and delta > self.accumulation_threshold:
            return None

        # Veto LONG when delta < -accumulation_threshold (strong institutional distribution)
        if direction_sign == 1 and delta < -self.accumulation_threshold:
            return None

        return signal


class SocialMomentumStrategy(Strategy):
    """
    Standalone strategy trading social media mention momentum with blow-off detection.

    Reads context["social_mentions"][symbol] = {count, z_score, sentiment}.

    Logic:
      - If z_score <= 3 or sentiment <= 0: no signal (return None)
      - If z_score > 3 and sentiment > 0:
        * Compute 5-day trailing price return
        * If 5-day return <= +20%: trade LONG (crowd inflow momentum)
        * If 5-day return > +20%: trade SHORT (fade blow-off top)
      - Gate on Kronos agreement: dist.stats["close"]["mean"] direction must
        align with chosen direction (LONG when mean > price, SHORT when mean < price)
      - Bracket: stop=pct_15, target=pct_85 for LONG; reversed for SHORT
      - Size: min(kelly_fraction * 0.5, 1.0)
      - Confidence: min(z_score / 6, 1.0)
      - Gracefully degrades: missing context key or symbol → None

    Args:
        stop_pct: Percentile for stop loss (default 15)
        target_pct: Percentile for target (default 85)
    """
    name = "social_momentum"

    def __init__(self, stop_pct: float = 15.0, target_pct: float = 85.0):
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    def _trailing_5d_return(self, history) -> float:
        """
        Compute 5-day trailing price return.
        Returns: (last_close - fifth_last_close) / fifth_last_close
        If fewer than 5 bars available, return 0.0.
        """
        closes = history["close"].values
        if len(closes) < 5:
            return 0.0
        last_close = closes[-1]
        fifth_last_close = closes[-5]
        if fifth_last_close == 0:
            return 0.0
        return (last_close - fifth_last_close) / fifth_last_close

    def generate_signal(self, dist, current_price, history, context):
        # Graceful degradation: if social_mentions context missing, return None
        social_dict = context.get("social_mentions")
        if social_dict is None:
            return None

        symbol = context.get("symbol")
        if symbol is None or symbol not in social_dict:
            return None

        mention_data = social_dict[symbol]

        # Extract z_score and sentiment
        z_score = mention_data.get("z_score")
        sentiment = mention_data.get("sentiment")

        # Missing required fields → None
        if z_score is None or sentiment is None:
            return None

        # z_score must exceed 3 and sentiment must be positive
        if z_score <= 3.0 or sentiment <= 0.0:
            return None

        # Compute 5-day trailing return
        trailing_5d_return = self._trailing_5d_return(history)

        # Determine direction: LONG for normal momentum, SHORT for blow-off fade
        if trailing_5d_return > 0.20:  # > +20%
            # Blow-off top: fade with SHORT
            chosen_direction = Direction.SHORT
        else:
            # Normal momentum: LONG on crowd inflow
            chosen_direction = Direction.LONG

        # Get Kronos distribution stats for agreement check
        s = dist.stats.get("close", {})
        mean_price = s.get("mean")

        # Gate on Kronos agreement
        if mean_price is None:
            return None

        # LONG requires mean > current_price; SHORT requires mean < current_price
        if chosen_direction == Direction.LONG:
            if mean_price <= current_price:
                return None  # Kronos disagrees: predicts down, but we want long
        else:  # SHORT
            if mean_price >= current_price:
                return None  # Kronos disagrees: predicts up, but we want short

        # Compute stop and target using percentiles (reversed for SHORT)
        if chosen_direction == Direction.LONG:
            stop = s.get(f"pct_{int(self.stop_pct)}")
            target = s.get(f"pct_{int(self.target_pct)}")
        else:  # SHORT
            stop = s.get(f"pct_{int(self.target_pct)}")
            target = s.get(f"pct_{int(self.stop_pct)}")

        if stop is None or target is None:
            return None

        # Compute Kelly fraction and size
        kelly = dist.kelly_fraction(current_price, target, stop)
        size = min(kelly * 0.5, 1.0)

        # Compute confidence from z_score, capped at 1.0
        confidence = min(z_score / 6.0, 1.0)

        # Compute expected value
        ev = dist.expected_value(current_price, target, stop)

        return Signal(
            direction=chosen_direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={
                "z_score": z_score,
                "sentiment": sentiment,
                "trailing_5d_return": trailing_5d_return,
                "momentum_type": "fade_blowoff" if trailing_5d_return > 0.20 else "crowd_inflow",
            }
        )


class EconCalendarGuardStrategy(Strategy):
    """
    Filters signals based on economic calendar events.

    Wraps a base strategy and applies economic calendar filtering:
      - Vetoes new entries when HIGH-impact events fall within the guard window
        (current_date, current_date + guard_days] (tomorrow through guard_days ahead).
      - Exposes should_tighten_stops(context) helper for orchestrator to consult
        regarding open positions in the guard window.
      - Gracefully degrades: if context["econ_events"] or context["current_date"]
        is missing, returns base signal unchanged.
      - Handles string ("YYYY-MM-DD"), date, and datetime objects for date parsing.

    Args:
        base_strategy: The wrapped strategy to filter.
        guard_days: Number of days ahead to monitor for high-impact events.
                   Default 1 (tomorrow only).
    """
    name = "econ_calendar_guard"

    def __init__(self, base_strategy: Strategy, guard_days: int = 1):
        self.base_strategy = base_strategy
        self.guard_days = guard_days

    def _parse_date(self, date_obj):
        """
        Convert various date formats to date object.
        Handles: str (YYYY-MM-DD), datetime.date, datetime.datetime.
        Returns: datetime.date or None if parsing fails.
        """
        if isinstance(date_obj, str):
            try:
                return datetime.strptime(date_obj, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                return None
        elif isinstance(date_obj, datetime):
            return date_obj.date()
        elif isinstance(date_obj, date):
            return date_obj
        return None

    def should_tighten_stops(self, context):
        """
        Check if any HIGH-impact event falls within the guard window.
        Returns True if stops should be tightened for open positions.

        Args:
            context: Dict containing "econ_events" and "current_date" keys.

        Returns:
            bool: True if in guard window, False otherwise.
                  Returns False if context keys are missing.
        """
        # Graceful degradation: missing context keys
        econ_events = context.get("econ_events")
        current_date_obj = context.get("current_date")

        if econ_events is None or current_date_obj is None:
            return False

        # Parse current_date
        current_date = self._parse_date(current_date_obj)
        if current_date is None:
            return False

        # Check for HIGH-impact events in guard window
        guard_end = current_date + timedelta(days=self.guard_days)

        for event in econ_events:
            event_date = self._parse_date(event.get("date"))
            impact = event.get("impact", "").lower()

            if event_date is None:
                continue

            # Check if event is within guard window: (current_date, current_date + guard_days]
            if current_date < event_date <= guard_end and impact == "high":
                return True

        return False

    def generate_signal(self, dist, current_price, history, context):
        # Get base signal from wrapped strategy
        signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if signal is None:
            return None

        # Graceful degradation: if econ context missing, return base signal unchanged
        econ_events = context.get("econ_events")
        if econ_events is None:
            return signal

        current_date_obj = context.get("current_date")
        if current_date_obj is None:
            return signal

        # Parse current_date
        current_date = self._parse_date(current_date_obj)
        if current_date is None:
            return signal

        # Check for HIGH-impact events in guard window
        # Guard window is (current_date, current_date + guard_days] (tomorrow through guard_days ahead)
        guard_end = current_date + timedelta(days=self.guard_days)

        for event in econ_events:
            event_date = self._parse_date(event.get("date"))
            impact = event.get("impact", "").lower()

            if event_date is None:
                continue

            # Veto new entries when HIGH-impact event is in guard window
            if current_date < event_date <= guard_end and impact == "high":
                return None

        return signal
