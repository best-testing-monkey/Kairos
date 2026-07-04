"""
kairos_forex.py
===============
Forex-specific strategies (2.1 – 2.10) for the Kairos framework.

All strategies read context fields documented in EXTENDED_STRATEGIES.md §6.1.
Missing context fields cause the strategy to return None gracefully.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from typing import Optional, Dict, List
from kairos_backtest import KairosDistribution, Direction, Signal, Strategy


# =============================================================================
# 2.1  Carry Trade
# =============================================================================

class CarryTrade(Strategy):
    """
    Long the high-yield currency, short the low-yield.
    Enters when carry-to-risk ratio (interest differential / predicted range)
    exceeds min_ratio.
    """
    name = "carry_trade"

    def __init__(self, min_ratio: float = 0.5, max_size: float = 0.3):
        self.min_ratio = min_ratio
        self.max_size = max_size

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        base_rate = context.get("base_interest_rate")
        quote_rate = context.get("quote_interest_rate")
        if base_rate is None or quote_rate is None:
            return None

        differential = base_rate - quote_rate
        if differential == 0:
            return None

        s = dist.stats["close"]
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price if current_price > 0 else 1.0
        if pred_range <= 0:
            return None

        carry_to_risk = abs(differential) / pred_range
        if carry_to_risk < self.min_ratio:
            return None

        direction = Direction.LONG if differential > 0 else Direction.SHORT
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(carry_to_risk * 0.1, self.max_size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(carry_to_risk / (self.min_ratio * 2), 1.0),
            expected_value=ev,
            metadata={"differential": differential, "carry_to_risk": carry_to_risk},
        )


# =============================================================================
# 2.2  Session Breakout
# =============================================================================

class SessionBreakout(Strategy):
    """
    Enters a breakout when predicted range is significantly wider than the
    Asian session range, in the direction of the predicted close.
    """
    name = "session_breakout"

    def __init__(self, range_multiplier: float = 1.5):
        self.multiplier = range_multiplier

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        asian_range = context.get("asian_session_range")
        if asian_range is None or asian_range <= 0:
            return None

        s_close = dist.stats["close"]
        s_high = dist.stats["high"]
        s_low = dist.stats["low"]
        pred_range = s_close["pct_90"] - s_close["pct_10"]

        if pred_range < asian_range * self.multiplier:
            return None

        if s_close["mean"] > current_price:
            direction = Direction.LONG
            stop = s_low["pct_10"]
            target = s_high["pct_90"]
        elif s_close["mean"] < current_price:
            direction = Direction.SHORT
            stop = s_high["pct_90"]
            target = s_low["pct_10"]
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        confidence = min(pred_range / (asian_range * self.multiplier) - 1.0, 1.0)

        return Signal(
            direction=direction,
            size=min(confidence * 0.4, 0.35),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"asian_range": asian_range, "pred_range": pred_range,
                      "range_multiplier": pred_range / asian_range},
        )


# =============================================================================
# 2.3  London Fix Fade
# =============================================================================

class LondonFixFade(Strategy):
    """
    Fades the 4pm London WM/Reuters fix move in direction of predicted close.
    """
    name = "london_fix_fade"

    def __init__(self, fix_time: str = "16:00", fade_threshold: float = 0.002):
        self.fix_time = fix_time
        self.threshold = fade_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        fix_price = context.get("fix_price")
        if fix_price is None or fix_price <= 0:
            return None

        pred_mean = dist.stats["close"]["mean"]

        if pred_mean < fix_price * (1 - self.threshold):
            direction = Direction.SHORT
            stop = fix_price * 1.01
            target = pred_mean
        elif pred_mean > fix_price * (1 + self.threshold):
            direction = Direction.LONG
            stop = fix_price * 0.99
            target = pred_mean
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        deviation = abs(fix_price - pred_mean) / fix_price
        confidence = min(deviation / self.threshold, 1.0)

        return Signal(
            direction=direction,
            size=min(confidence * 0.3, 0.3),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"fix_price": fix_price, "pred_mean": pred_mean,
                      "deviation": deviation},
        )


# =============================================================================
# 2.4  Central Bank Divergence
# =============================================================================

class CBDivergence(Strategy):
    """
    Trades the spread between central bank policy rates.
    Requires Kairos predicted Sharpe > 0.5 for entry.
    """
    name = "cb_divergence"

    def __init__(self, max_size: float = 0.4):
        self.max_size = max_size

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        base_rate = context.get("base_cb_rate")
        quote_rate = context.get("quote_cb_rate")
        if base_rate is None or quote_rate is None:
            return None

        divergence = base_rate - quote_rate
        pred_sharpe = dist.predicted_sharpe()

        if abs(pred_sharpe) <= 0.5:
            return None

        if divergence > 0 and pred_sharpe > 0.5:
            direction = Direction.LONG
        elif divergence < 0 and pred_sharpe > 0.5:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(abs(divergence) * pred_sharpe, self.max_size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(pred_sharpe) * 0.5, 1.0),
            expected_value=ev,
            metadata={"divergence": divergence, "pred_sharpe": pred_sharpe},
        )


# =============================================================================
# 2.5  Safe Haven Rotation
# =============================================================================

class SafeHavenRotation(Strategy):
    """
    Cross-asset regime rotation: long safe havens vs. risk assets.
    Classifies assets by symbol suffix or context["safe_havens"] list.
    """
    name = "safe_haven_rotation"

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        multi_preds = context.get("multi_asset_predictions")
        safe_havens: List[str] = context.get("safe_havens", ["JPY", "CHF", "USD"])
        risk_assets: List[str] = context.get("risk_assets", ["AUD", "NZD", "TRY"])

        if not multi_preds:
            return None

        safe_sharpes, risk_sharpes = [], []
        for sym, pred in multi_preds.items():
            sym_upper = sym.upper()
            ps = pred.dist.predicted_sharpe()
            if any(sh in sym_upper for sh in safe_havens):
                safe_sharpes.append(ps)
            elif any(ra in sym_upper for ra in risk_assets):
                risk_sharpes.append(ps)

        if not safe_sharpes or not risk_sharpes:
            return None

        safe_mean = float(np.mean(safe_sharpes))
        risk_mean = float(np.mean(risk_sharpes))
        gap = safe_mean - risk_mean

        if gap > self.threshold:
            direction = Direction.LONG
        elif -gap > self.threshold:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=min(abs(gap) * 0.3, 0.35),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(gap) / self.threshold * 0.5, 1.0),
            expected_value=ev,
            metadata={"safe_sharpe": safe_mean, "risk_sharpe": risk_mean, "gap": gap},
        )


# =============================================================================
# 2.6  Triangular Arbitrage
# =============================================================================

class TriangularArbitrage(Strategy):
    """
    Detects cross-rate inefficiencies: EUR/USD * USD/JPY ≠ EUR/JPY.
    Requires context["leg_dists"] = dict of KairosDistribution per leg.
    """
    name = "triangular_arbitrage"

    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        leg_dists: Optional[Dict[str, KairosDistribution]] = context.get("leg_dists")
        if leg_dists is None or len(leg_dists) < 3:
            return None

        legs = list(leg_dists.items())
        # Expect three legs; compute synthetic cross from first two, compare to third
        _, dist_ab = legs[0]
        _, dist_bc = legs[1]
        _, dist_ac = legs[2]

        pred_ab = dist_ab.stats["close"]["mean"]
        pred_bc = dist_bc.stats["close"]["mean"]
        pred_ac = dist_ac.stats["close"]["mean"]

        if pred_ab <= 0 or pred_bc <= 0 or pred_ac <= 0:
            return None

        synthetic = pred_ab * pred_bc
        deviation = abs(synthetic - pred_ac) / pred_ac

        if deviation <= self.threshold:
            return None

        return Signal(
            direction=Direction.FLAT,
            size=min(deviation * 10, 0.25),
            entry=current_price,
            stop=current_price * (1 - deviation),
            target=current_price * (1 + deviation),
            strategy_name=self.name,
            confidence=min(deviation / self.threshold * 0.5, 1.0),
            expected_value=deviation * current_price,
            metadata={"action": "tri_arb", "synthetic": synthetic,
                      "actual": pred_ac, "deviation": deviation},
        )


# =============================================================================
# 2.7  Sovereign CDS Spread Filter
# =============================================================================

class CDSSpreadFilter(Strategy):
    """
    Blocks signals that conflict with sovereign CDS spread direction.
    Widening CDS blocks LONG; tightening CDS blocks SHORT.
    """
    name = "cds_spread_filter"

    def __init__(self, base_strategy: Strategy):
        self.base = base_strategy

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        cds_change = context.get("cds_spread_change", 0.0)

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        if cds_change > 0 and sig.direction == Direction.LONG:
            return None
        if cds_change < 0 and sig.direction == Direction.SHORT:
            return None

        sig.strategy_name = self.name
        return sig


# =============================================================================
# 2.8  CFTC COT Positioning Filter
# =============================================================================

class COTPositioningFilter(Strategy):
    """
    Contrarian filter: blocks LONG when speculators are extremely long,
    blocks SHORT when speculators are extremely short.
    spec_position normalized 0-1 (1 = max long).
    """
    name = "cot_positioning_filter"

    def __init__(self, base_strategy: Strategy, extreme_threshold: float = 0.8):
        self.base = base_strategy
        self.threshold = extreme_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        spec_pos = context.get("speculator_net_position")
        if spec_pos is None:
            sig = self.base.generate_signal(dist, current_price, history, context)
            if sig:
                sig.strategy_name = self.name
            return sig

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        if spec_pos > self.threshold and sig.direction == Direction.LONG:
            return None
        if spec_pos < (1 - self.threshold) and sig.direction == Direction.SHORT:
            return None

        sig.strategy_name = self.name
        return sig


# =============================================================================
# 2.9  Asian Range Breakout
# =============================================================================

class AsianRangeBreakout(Strategy):
    """
    Enters breakout when predicted high/low extends outside Asian session range.
    """
    name = "asian_range_breakout"

    def __init__(self, confirmation_pct: float = 0.001):
        self.confirmation = confirmation_pct

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        asian_high = context.get("asian_high")
        asian_low = context.get("asian_low")
        if asian_high is None or asian_low is None:
            return None

        pred_high = dist.stats["high"]["mean"]
        pred_low = dist.stats["low"]["mean"]

        if pred_high > asian_high * (1 + self.confirmation):
            direction = Direction.LONG
            entry = asian_high
            stop = asian_low
            target = pred_high
        elif pred_low < asian_low * (1 - self.confirmation):
            direction = Direction.SHORT
            entry = asian_low
            stop = asian_high
            target = pred_low
        else:
            return None

        if stop <= 0 or target <= 0:
            return None

        ev = dist.expected_value(entry, target, stop)
        if ev <= 0:
            return None

        breakout_pct = abs(pred_high - asian_high) / asian_high if direction == Direction.LONG \
            else abs(asian_low - pred_low) / asian_low

        return Signal(
            direction=direction,
            size=min(breakout_pct * 10, 0.35),
            entry=entry,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(breakout_pct / self.confirmation, 1.0),
            expected_value=ev,
            metadata={"asian_high": asian_high, "asian_low": asian_low,
                      "breakout_pct": breakout_pct},
        )


# =============================================================================
# 2.10  OIS Swap Spread (Interest Rate Swap Spread)
# =============================================================================

class OISSwapSpread(Strategy):
    """
    Trades FX direction when OIS curve slope aligns with Kairos close prediction.
    Steepening curve = hawkish = currency appreciation.
    """
    name = "ois_swap_spread"

    def __init__(self, max_size: float = 0.3):
        self.max_size = max_size

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        ois_curve: Optional[List[float]] = context.get("ois_curve")
        if not ois_curve or len(ois_curve) < 2:
            return None

        curve_slope = ois_curve[-1] - ois_curve[0]
        pred_mean = dist.stats["close"]["mean"]

        if curve_slope > 0 and pred_mean > current_price:
            direction = Direction.LONG
        elif curve_slope < 0 and pred_mean < current_price:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(abs(curve_slope) * 10, self.max_size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(curve_slope) * 5, 1.0),
            expected_value=ev,
            metadata={"curve_slope": curve_slope, "ois_front": ois_curve[0],
                      "ois_back": ois_curve[-1]},
        )
