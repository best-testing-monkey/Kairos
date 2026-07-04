"""
kairos_stocks.py
================
Stock-specific strategies (3.1 – 3.12) for the Kairos framework.

All strategies read context fields documented in EXTENDED_STRATEGIES.md §6.1.
Missing context fields cause the strategy to return None gracefully.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from typing import Optional, Dict
from kairos_backtest import KairosDistribution, Direction, Signal, Strategy


# =============================================================================
# 3.1  Post-Earnings Announcement Drift (PEAD)
# =============================================================================

class PEAD(Strategy):
    """
    Trades the earnings surprise drift.  Enters when SUE is extreme and
    Kairos confirms the direction.  Holds for hold_days (tracked via metadata).
    """
    name = "pead"

    def __init__(self, min_sue: float = 1.5, max_size: float = 0.3,
                 hold_days: int = 20):
        self.min_sue = min_sue
        self.max_size = max_size
        self.hold_days = hold_days

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        sue = context.get("standardized_unexpected_earnings")
        if sue is None or abs(sue) < self.min_sue:
            return None

        s = dist.stats["close"]
        pred_mean = s["mean"]

        if sue > 0 and pred_mean > current_price:
            direction = Direction.LONG
        elif sue < 0 and pred_mean < current_price:
            direction = Direction.SHORT
        else:
            return None

        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(abs(sue) * 0.15, self.max_size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(sue) / (self.min_sue * 2), 1.0),
            expected_value=ev,
            metadata={"sue": sue, "hold_days": self.hold_days},
        )


# =============================================================================
# 3.2  Earnings Momentum (SUE + EAR)
# =============================================================================

class EarningsMomentum(Strategy):
    """
    Composite earnings signal: SUE * 0.6 + EAR * 0.4.
    Enters when composite exceeds threshold and Kairos confirms direction.
    """
    name = "earnings_momentum"

    def __init__(self, threshold: float = 1.0, max_size: float = 0.3):
        self.threshold = threshold
        self.max_size = max_size

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        sue = context.get("sue")
        ear = context.get("ear")
        if sue is None or ear is None:
            return None

        composite = sue * 0.6 + ear * 0.4
        if abs(composite) < self.threshold:
            return None

        direction = Direction.LONG if composite > 0 else Direction.SHORT
        pred_sharpe = dist.predicted_sharpe()

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(abs(composite) * abs(pred_sharpe) * 0.2, self.max_size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(composite) / (self.threshold * 3), 1.0),
            expected_value=ev,
            metadata={"composite": composite, "sue": sue, "ear": ear,
                      "pred_sharpe": pred_sharpe},
        )


# =============================================================================
# 3.3  Index Rebalancing Arbitrage
# =============================================================================

class IndexRebalance(Strategy):
    """
    Front-runs index fund forced buying (addition) or selling (deletion).
    High conviction: size = 0.5 for index events.
    """
    name = "index_rebalance"

    def __init__(self, event_window_days: int = 3):
        self.window = event_window_days

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        event = context.get("index_event")
        if event not in ("addition", "deletion"):
            return None

        s = dist.stats["close"]
        pred_mean = s["mean"]

        if event == "addition" and pred_mean > current_price:
            direction = Direction.LONG
        elif event == "deletion" and pred_mean < current_price:
            direction = Direction.SHORT
        else:
            return None

        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=0.5,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=0.8,
            expected_value=ev,
            metadata={"index_event": event, "event_window": self.window},
        )


# =============================================================================
# 3.4  Sector Rotation Momentum
# =============================================================================

class SectorRotation(Strategy):
    """
    Ranks sector ETFs by predicted Sharpe.  Long top sectors, signal for current
    symbol based on its relative ranking.
    """
    name = "sector_rotation"

    def __init__(self, top_n: int = 2, bottom_n: int = 2):
        self.top_n = top_n
        self.bottom_n = bottom_n

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        sector_preds = context.get("sector_predictions")
        if not sector_preds:
            return None

        rankings = sorted(
            [(sym, pred.dist.predicted_sharpe()) for sym, pred in sector_preds.items()],
            key=lambda x: x[1], reverse=True
        )
        if len(rankings) < self.top_n + self.bottom_n:
            return None

        top_syms = {s for s, _ in rankings[:self.top_n]}
        bot_syms = {s for s, _ in rankings[-self.bottom_n:]}
        current_sym = context.get("symbol", "")

        if current_sym in top_syms:
            direction = Direction.LONG
        elif current_sym in bot_syms:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        rank_pos = next(i for i, (sym, _) in enumerate(rankings) if sym == current_sym)
        n = len(rankings)
        normalized_rank = rank_pos / max(n - 1, 1)
        confidence = abs(0.5 - normalized_rank) * 2

        return Signal(
            direction=direction,
            size=min(confidence * 0.4, 0.35),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"rank": rank_pos, "n_sectors": n},
        )


# =============================================================================
# 3.5  Pairs Trading (Cointegration)
# =============================================================================

class CointegrationPairs(Strategy):
    """
    Trades mean reversion of a cointegrated spread.
    Requires context["spread_dist"] = KairosDistribution of the spread and
    context["current_spread"] = float.
    """
    name = "cointegration_pairs"

    def __init__(self, hedge_ratio: float = 1.0, pair_symbol: str = ""):
        self.hedge_ratio = hedge_ratio
        self.pair_symbol = pair_symbol

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        spread_dist: Optional[KairosDistribution] = context.get("spread_dist")
        current_spread = context.get("current_spread")
        if spread_dist is None or current_spread is None:
            return None

        s = spread_dist.stats["close"]
        pred_mean = s["mean"]

        if pred_mean > current_spread:
            direction = Direction.SHORT
            stop = s["pct_90"]
            target = s["pct_10"]
        elif pred_mean < current_spread:
            direction = Direction.LONG
            stop = s["pct_10"]
            target = s["pct_90"]
        else:
            return None

        ev = spread_dist.expected_value(current_spread, target, stop)
        if ev <= 0:
            return None

        z_score = (current_spread - s["mean"]) / max(s["std"], 1e-9)

        return Signal(
            direction=direction,
            size=min(abs(z_score) * 0.1, 0.3),
            entry=current_price,
            stop=current_price * (1 + (stop - current_spread) / max(current_spread, 1e-9)),
            target=current_price * (1 + (target - current_spread) / max(current_spread, 1e-9)),
            strategy_name=self.name,
            confidence=min(abs(z_score) / 3, 1.0),
            expected_value=ev,
            metadata={"pair": self.pair_symbol, "hedge_ratio": self.hedge_ratio,
                      "z_score": z_score, "current_spread": current_spread},
        )


# =============================================================================
# 3.6  Merger Arbitrage
# =============================================================================

class MergerArb(Strategy):
    """
    Long the target when predicted close is near the offer price and
    deal probability is high.
    """
    name = "merger_arb"

    def __init__(self, min_deal_prob: float = 0.8):
        self.min_deal_prob = min_deal_prob

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        offer_price = context.get("offer_price")
        deal_prob = context.get("deal_probability", 0.0)
        if offer_price is None or deal_prob < self.min_deal_prob:
            return None

        pred_mean = dist.stats["close"]["mean"]
        if pred_mean < offer_price * 0.98:
            return None

        stop = current_price * 0.95
        target = offer_price

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = deal_prob * 0.5

        return Signal(
            direction=Direction.LONG,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=deal_prob,
            expected_value=ev,
            metadata={"offer_price": offer_price, "deal_prob": deal_prob,
                      "spread": (offer_price - current_price) / current_price},
        )


# =============================================================================
# 3.7  Buyback Yield Capture
# =============================================================================

class BuybackYield(Strategy):
    """
    Longs near a buyback-supported price floor when predicted close is above
    the floor.
    """
    name = "buyback_yield"

    def __init__(self, proximity: float = 0.02):
        self.proximity = proximity

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        buyback_floor = context.get("buyback_floor")
        if buyback_floor is None or buyback_floor <= 0:
            return None

        s = dist.stats["low"]
        pred_low = s["mean"]
        pred_close = dist.stats["close"]["mean"]

        if abs(pred_low - buyback_floor) / buyback_floor > self.proximity:
            return None
        if pred_close <= buyback_floor:
            return None

        entry = pred_low
        stop = buyback_floor * 0.98
        target = pred_close

        ev = dist.expected_value(entry, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=Direction.LONG,
            size=0.3,
            entry=entry,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=0.7,
            expected_value=ev,
            metadata={"buyback_floor": buyback_floor, "pred_low": pred_low},
        )


# =============================================================================
# 3.8  Short Interest Squeeze
# =============================================================================

class ShortSqueeze(Strategy):
    """
    Amplifies long signals when short interest is high and predicted Sharpe > 1.
    """
    name = "short_squeeze"

    def __init__(self, base_strategy: Strategy,
                 min_short_interest: float = 0.15,
                 min_sharpe: float = 1.0,
                 max_size: float = 0.6):
        self.base = base_strategy
        self.min_si = min_short_interest
        self.min_sharpe = min_sharpe
        self.max_size = max_size

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        si = context.get("short_interest_ratio", 0.0)
        if si < self.min_si:
            return None

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None or sig.direction != Direction.LONG:
            return None

        pred_sharpe = dist.predicted_sharpe()
        if pred_sharpe < self.min_sharpe:
            return None

        sig.size = min(sig.size * 1.5, self.max_size)
        sig.metadata["squeeze_potential"] = si * pred_sharpe
        sig.strategy_name = self.name
        return sig


# =============================================================================
# 3.9  Insider Transaction Clustering
# =============================================================================

class InsiderCluster(Strategy):
    """
    Passes signals only when insider activity aligns.
    insider_signal: +1 buying cluster, -1 selling cluster, 0 neutral.
    Amplifies longs by 20% when insiders are buying.
    """
    name = "insider_cluster"

    def __init__(self, base_strategy: Strategy):
        self.base = base_strategy

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        insider_signal = context.get("insider_signal", 0)

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        if insider_signal == -1 and sig.direction == Direction.LONG:
            return None
        if insider_signal == 1 and sig.direction == Direction.SHORT:
            return None
        if insider_signal == 1:
            sig.size = min(sig.size * 1.2, 1.0)

        sig.strategy_name = self.name
        return sig


# =============================================================================
# 3.10  Dark Pool Print Analysis
# =============================================================================

class DarkPoolFilter(Strategy):
    """
    Filters signals by dark pool sentiment.
    dark_pool_sentiment: -1 (bearish) to +1 (bullish).
    """
    name = "dark_pool_filter"

    def __init__(self, base_strategy: Strategy, sentiment_threshold: float = 0.3):
        self.base = base_strategy
        self.threshold = sentiment_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        sentiment = context.get("dark_pool_sentiment", 0.0)

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        if sentiment < -self.threshold and sig.direction == Direction.LONG:
            return None
        if sentiment > self.threshold and sig.direction == Direction.SHORT:
            return None

        sig.strategy_name = self.name
        return sig


# =============================================================================
# 3.11  Share Buyback Announcement Drift
# =============================================================================

class BuybackDrift(Strategy):
    """
    Enters long within drift_window days of a buyback announcement when
    predicted close is above current price.
    """
    name = "buyback_drift"

    def __init__(self, max_size: float = 0.3, drift_window: int = 5):
        self.max_size = max_size
        self.window = drift_window

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        announcement_date = context.get("buyback_announcement_date")
        current_date = context.get("date")
        if announcement_date is None or current_date is None:
            return None

        try:
            days_since = (pd.Timestamp(current_date) - pd.Timestamp(announcement_date)).days
        except Exception:
            return None

        if days_since < 0 or days_since > self.window:
            return None

        s = dist.stats["close"]
        pred_mean = s["mean"]
        if pred_mean <= current_price:
            return None

        target = pred_mean
        stop = current_price * 0.97

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=Direction.LONG,
            size=min(0.3, self.max_size),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=0.65,
            expected_value=ev,
            metadata={"days_since_announcement": days_since, "window": self.window},
        )


# =============================================================================
# 3.12  Dividend Capture
# =============================================================================

class DividendCapture(Strategy):
    """
    Buys 1-5 days before ex-dividend date when predicted close shows
    sufficient recovery after the dividend drop.
    """
    name = "dividend_capture"

    def __init__(self, min_recovery: float = 0.5):
        self.min_recovery = min_recovery

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        dividend = context.get("dividend_amount")
        ex_div_date = context.get("ex_div_date")
        current_date = context.get("date")
        if dividend is None or ex_div_date is None or current_date is None:
            return None

        try:
            days_to_ex = (pd.Timestamp(ex_div_date) - pd.Timestamp(current_date)).days
        except Exception:
            return None

        if days_to_ex < 1 or days_to_ex > 5:
            return None

        pred_close = dist.stats["close"]["mean"]
        if pred_close <= current_price + dividend * self.min_recovery:
            return None

        target = pred_close
        stop = current_price - dividend

        if stop <= 0:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=Direction.LONG,
            size=0.25,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=0.6,
            expected_value=ev,
            metadata={"dividend": dividend, "days_to_ex": days_to_ex,
                      "ex_div_date": str(ex_div_date)},
        )
