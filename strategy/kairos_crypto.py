"""
kairos_crypto.py
================
Crypto-specific strategies (1.1 – 1.10) for the Kairos framework.

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
# 1.1  Funding Rate Arbitrage
# =============================================================================

class FundingRateArbitrage(Strategy):
    """
    Long spot + short perp when funding rate is positive and volatility is low.
    Returns FLAT signal with metadata: action="long_spot_short_perp".
    """
    name = "funding_rate_arbitrage"

    def __init__(self, min_funding_threshold: float = 0.0001,
                 max_volatility_threshold: float = 0.02,
                 max_position_size: float = 0.3):
        self.min_funding = min_funding_threshold
        self.max_vol = max_volatility_threshold
        self.max_size = max_position_size

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        funding_rate = context.get("funding_rate")
        if funding_rate is None or funding_rate < self.min_funding:
            return None

        s = dist.stats["close"]
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price if current_price > 0 else 1.0
        if pred_range > self.max_vol:
            return None

        if s["mean"] < current_price:
            return None

        size = min(funding_rate / max(pred_range, 1e-9), self.max_size)

        return Signal(
            direction=Direction.FLAT,
            size=size,
            entry=current_price,
            stop=current_price * (1 - self.max_vol),
            target=current_price * (1 + funding_rate * 8),
            strategy_name=self.name,
            confidence=min(funding_rate / self.min_funding * 0.5, 1.0),
            expected_value=funding_rate * current_price,
            metadata={"action": "long_spot_short_perp",
                      "funding_rate": funding_rate,
                      "predicted_range": pred_range},
        )


# =============================================================================
# 1.2  Basis Trade
# =============================================================================

class BasisTrade(Strategy):
    """
    Short futures premium when it is wide vs. spot; long spot.
    Returns FLAT signal with metadata: action="short_basis".
    """
    name = "basis_trade"

    def __init__(self, min_basis: float = 0.005, max_basis: float = 0.15):
        self.min_basis = min_basis
        self.max_basis = max_basis

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        futures_mark = context.get("futures_mark_price")
        if futures_mark is None or current_price <= 0:
            return None

        basis = (futures_mark - current_price) / current_price
        if basis < self.min_basis or basis > self.max_basis:
            return None

        s = dist.stats["close"]
        predicted_close = s["mean"]
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price

        if predicted_close > futures_mark * 0.995:
            return None

        confidence = min(basis / max(pred_range, 1e-9), 1.0)

        return Signal(
            direction=Direction.FLAT,
            size=min(basis * 2, 0.3),
            entry=current_price,
            stop=futures_mark * 1.02,
            target=current_price,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=basis * current_price,
            metadata={"action": "short_basis", "basis": basis,
                      "futures_mark": futures_mark},
        )


# =============================================================================
# 1.3  Stablecoin Depeg
# =============================================================================

class StablecoinDepeg(Strategy):
    """
    Mean-reversion to $1 peg when stablecoin is off-peg.
    Only enters if predicted distribution is inside peg band.
    """
    name = "stablecoin_depeg"

    def __init__(self, lower_peg: float = 0.995,
                 upper_peg: float = 1.005,
                 max_deviation: float = 0.05):
        self.lower_peg = lower_peg
        self.upper_peg = upper_peg
        self.max_deviation = max_deviation

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        if current_price < self.lower_peg - self.max_deviation:
            return None
        if current_price > self.upper_peg + self.max_deviation:
            return None

        if self.lower_peg <= current_price <= self.upper_peg:
            return None

        if current_price < self.lower_peg:
            direction = Direction.LONG
            target = 1.0
            stop = current_price * 0.99
        else:
            direction = Direction.SHORT
            target = 1.0
            stop = current_price * 1.01

        pred_mean = dist.stats["close"]["mean"]
        if not (self.lower_peg <= pred_mean <= self.upper_peg):
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=0.4,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=0.85,
            expected_value=ev,
            metadata={"depeg_size": abs(current_price - 1.0)},
        )


# =============================================================================
# 1.4  Exchange Spread Arbitrage
# =============================================================================

class ExchangeSpreadArbitrage(Strategy):
    """
    Cross-exchange arbitrage using predicted distributions from two exchanges.
    Returns FLAT signal when a risk-free spread exists.
    """
    name = "exchange_spread"

    def __init__(self, fee_buffer: float = 0.002):
        self.fee_buffer = fee_buffer

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        other_dist: Optional[KairosDistribution] = context.get("other_exchange_dist")
        other_price = context.get("other_exchange_price")
        if other_dist is None or other_price is None:
            return None

        pred_low_a = dist.stats["low"]["mean"]
        pred_high_b = other_dist.stats["high"]["mean"]
        pred_high_a = dist.stats["high"]["mean"]
        pred_low_b = other_dist.stats["low"]["mean"]

        if pred_low_a > pred_high_b * (1 + self.fee_buffer):
            action = "arb_A_to_B"
            spread = (pred_low_a - pred_high_b) / pred_high_b
        elif pred_high_b > pred_low_a * (1 + self.fee_buffer):
            action = "arb_B_to_A"
            spread = (pred_high_b - pred_low_a) / pred_low_a
        else:
            return None

        _ = pred_high_a  # referenced to avoid unused-variable lint
        _ = pred_low_b

        return Signal(
            direction=Direction.FLAT,
            size=min(spread * 10, 0.3),
            entry=current_price,
            stop=current_price * (1 - self.fee_buffer * 2),
            target=current_price * (1 + spread),
            strategy_name=self.name,
            confidence=min(spread / self.fee_buffer, 1.0),
            expected_value=spread * current_price,
            metadata={"action": action, "spread": spread},
        )


# =============================================================================
# 1.5  Liquidation Cluster Front-Run
# =============================================================================

class LiquidationFrontRun(Strategy):
    """
    Front-runs predicted wicks to visible liquidation walls.
    Enters in the direction of the predicted extreme, targeting the wall.
    """
    name = "liquidation_front_run"

    def __init__(self, proximity: float = 0.005):
        self.proximity = proximity

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        liq_walls: List[float] = context.get("liquidation_walls") or []
        if not liq_walls or current_price <= 0:
            return None

        pred_high = dist.stats["high"]["pct_90"]
        pred_low = dist.stats["low"]["pct_10"]

        for wall in liq_walls:
            if abs(pred_high - wall) / current_price < self.proximity:
                ev = dist.expected_value(current_price, wall, wall * 1.02)
                if ev <= 0:
                    continue
                return Signal(
                    direction=Direction.SHORT,
                    size=0.2,
                    entry=current_price,
                    stop=wall * 1.02,
                    target=wall,
                    strategy_name=self.name,
                    confidence=0.6,
                    expected_value=ev,
                    metadata={"wall": wall, "type": "high_wall"},
                )
            if abs(pred_low - wall) / current_price < self.proximity:
                ev = dist.expected_value(current_price, wall, wall * 0.98)
                if ev <= 0:
                    continue
                return Signal(
                    direction=Direction.LONG,
                    size=0.2,
                    entry=current_price,
                    stop=wall * 0.98,
                    target=wall,
                    strategy_name=self.name,
                    confidence=0.6,
                    expected_value=ev,
                    metadata={"wall": wall, "type": "low_wall"},
                )
        return None


# =============================================================================
# 1.6  Funding Rate Prediction
# =============================================================================

class FundingRatePrediction(Strategy):
    """
    Predicts next funding rate from the price path and positions early.
    Short perp if predicted mean is above current (positive funding incoming).
    Long perp if predicted mean is below current (negative funding incoming).
    """
    name = "funding_rate_prediction"

    def __init__(self, funding_threshold: float = 0.002):
        self.threshold = funding_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        s = dist.stats["close"]
        pred_mean = s["mean"]

        if pred_mean > current_price * (1 + self.threshold):
            direction = Direction.SHORT
            stop = dist.stats["high"]["pct_90"]
            target = current_price * 0.99
        elif pred_mean < current_price * (1 - self.threshold):
            direction = Direction.LONG
            stop = dist.stats["low"]["pct_10"]
            target = current_price * 1.01
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        deviation = abs(pred_mean - current_price) / current_price
        size = min(deviation / self.threshold * 0.2, 0.3)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(deviation / self.threshold * 0.5, 1.0),
            expected_value=ev,
            metadata={"predicted_funding": "positive" if direction == Direction.SHORT else "negative",
                      "pred_deviation": deviation},
        )


# =============================================================================
# 1.7  On-Chain Exchange Flow Filter
# =============================================================================

class OnChainFlowFilter(Strategy):
    """
    Wrapper: blocks signals that conflict with on-chain exchange flows.
    Large net_inflow (deposit to sell) blocks LONG; net_outflow blocks SHORT.
    """
    name = "onchain_flow_filter"

    def __init__(self, base_strategy: Strategy):
        self.base = base_strategy

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        inflow = context.get("exchange_inflow", 0.0)
        outflow = context.get("exchange_outflow", 0.0)
        net_flow = inflow - outflow

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None
        if sig.direction == Direction.LONG and net_flow > 0:
            return None
        if sig.direction == Direction.SHORT and net_flow < 0:
            return None

        sig.strategy_name = self.name
        return sig


# =============================================================================
# 1.8  Options Gamma Squeeze
# =============================================================================

class GammaSqueeze(Strategy):
    """
    Trades in the direction of a gamma squeeze toward a heavy-gamma strike.
    """
    name = "gamma_squeeze"

    def __init__(self, strike_proximity: float = 0.01):
        self.proximity = strike_proximity

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        gamma_map: Optional[Dict[float, float]] = context.get("gamma_by_strike")
        if not gamma_map or current_price <= 0:
            return None

        pred_mean = dist.stats["close"]["mean"]
        total_gamma = sum(gamma_map.values())
        if total_gamma <= 0:
            return None

        # Find the strike with maximum gamma near pred_mean
        best_strike = max(gamma_map.keys(),
                          key=lambda k: gamma_map[k] / (1 + abs(k - pred_mean) / current_price))
        max_gamma = gamma_map[best_strike]
        squeeze_intensity = max_gamma / total_gamma

        if pred_mean > current_price and best_strike > current_price:
            direction = Direction.LONG
            stop = dist.stats["low"]["pct_10"]
            target = best_strike
        elif pred_mean < current_price and best_strike < current_price:
            direction = Direction.SHORT
            stop = dist.stats["high"]["pct_90"]
            target = best_strike
        else:
            return None

        if abs(best_strike - current_price) / current_price > self.proximity * 5:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=min(squeeze_intensity, 0.4),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=squeeze_intensity,
            expected_value=ev,
            metadata={"squeeze_strike": best_strike, "squeeze_intensity": squeeze_intensity},
        )


# =============================================================================
# 1.9  Hash Rate Difficulty Filter
# =============================================================================

class HashRateFilter(Strategy):
    """
    BTC-specific filter: only passes LONG signals when hash rate is recovering
    (7-day MA > 30-day MA). Blocks all signals during miner capitulation.
    """
    name = "hash_rate_filter"

    def __init__(self, base_strategy: Strategy):
        self.base = base_strategy

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        ma7 = context.get("hash_rate_ma7")
        ma30 = context.get("hash_rate_ma30")

        if ma7 is None or ma30 is None:
            return None
        if ma7 < ma30:
            return None

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None
        if sig.direction == Direction.SHORT:
            return None

        sig.strategy_name = self.name
        return sig


# =============================================================================
# 1.10  Perp-Spot Funding Harvest
# =============================================================================

class FundingHarvest(Strategy):
    """
    Cross-asset carry: rotates into assets with highest funding yield
    relative to predicted volatility. Returns FLAT signal for delta-neutral hold.
    """
    name = "funding_harvest"

    def __init__(self, top_n: int = 3, min_carry_sharpe: float = 0.1):
        self.top_n = top_n
        self.min_carry_sharpe = min_carry_sharpe

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        multi_preds = context.get("multi_asset_predictions")
        funding_rates: Dict[str, float] = context.get("funding_rates") or {}

        if not multi_preds or not funding_rates:
            # Single-asset fallback
            symbol = context.get("symbol", "unknown")
            fr = funding_rates.get(symbol, 0.0)
            if fr <= 0:
                return None
            s = dist.stats["close"]
            pred_vol = s["std"] / current_price if current_price > 0 else 1.0
            carry_sharpe = fr / max(pred_vol, 1e-9)
            if carry_sharpe < self.min_carry_sharpe:
                return None
            return Signal(
                direction=Direction.FLAT,
                size=min(carry_sharpe * 0.1, 0.3),
                entry=current_price,
                stop=current_price * 0.97,
                target=current_price * (1 + fr * 24),
                strategy_name=self.name,
                confidence=min(carry_sharpe, 1.0),
                expected_value=fr * current_price,
                metadata={"action": "funding_harvest", "carry_sharpe": carry_sharpe},
            )

        ranked = []
        for sym, pred in multi_preds.items():
            fr = funding_rates.get(sym, 0.0)
            if fr <= 0:
                continue
            s = pred.dist.stats["close"]
            pred_vol = s["std"] / pred.current_price if pred.current_price > 0 else 1.0
            carry_sharpe = fr / max(pred_vol, 1e-9)
            ranked.append((sym, carry_sharpe, fr))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top = [(s, cs, fr) for s, cs, fr in ranked[:self.top_n]
               if cs >= self.min_carry_sharpe]
        if not top:
            return None

        best_sym, best_cs, best_fr = top[0]
        return Signal(
            direction=Direction.FLAT,
            size=min(best_cs * 0.1, 0.3),
            entry=current_price,
            stop=current_price * 0.97,
            target=current_price * (1 + best_fr * 24),
            strategy_name=self.name,
            confidence=min(best_cs, 1.0),
            expected_value=best_fr * current_price,
            metadata={"action": "funding_harvest",
                      "top_assets": [s for s, _, _ in top],
                      "carry_sharpe": best_cs},
        )
