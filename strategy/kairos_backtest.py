"""
kairos_backtest.py
==================
Kairos Distribution Backtesting Framework

A walk-forward backtester for crypto strategies using the Kairos
prediction distribution (60 Monte Carlo samples per bar).

Decision Tree
-------------
Kairos Distribution (60 samples per OHLC)
|
|-- [1] ENTROPY FILTER
|   High entropy (> threshold) --> NO TRADE
|   Low entropy --> Continue
|
|-- [2] BIMODALITY FILTER
|   Bimodal --> NO TRADE (event risk, wait for resolution)
|   Unimodal --> Continue
|
|-- [3] REGIME DETECTION (CV + Predicted Range / Current Price)
|   |
|   |-- RANGE DAY (tight distribution, narrow predicted range)
|   |   |-- RangeTradingStrategy
|   |   |-- PercentileEntryStrategy (mean reversion)
|   |   |-- FadeExtremeStrategy
|   |   |-- BollingerValidationStrategy
|   |   |-- HighLowStrategy (range-bound version)
|   |
|   |-- TREND DAY (wide distribution, strong directional bias)
|   |   |-- TrendFollowingStrategy
|   |   |-- SkewStrategy
|   |   |-- MomentumContinuationStrategy
|   |   |-- MACDFilterStrategy
|   |   |-- RSIFilterStrategy
|   |   |-- DynamicBracketStrategy
|   |
|   |-- UNCERTAIN
|       |-- NO TRADE
|
|-- [4] EXPECTED VALUE FILTER
|   EV <= 0 --> NO TRADE
|   EV > 0 --> Execute
|
|-- [5] POSITION SIZING
    |-- KellyCriterion (half-Kelly cap)
    |-- InverseVarianceSizing
    |-- ConfidenceWeighting

Usage
-----
    from kairos_backtest import (
        KairosPredictor, KairosDistribution, DecisionTreeRouter,
        BacktestEngine, Signal, Direction
    )

    def predict_kairos_cloud(df: pd.DataFrame) -> List[pd.DataFrame]:
        # Your 60-run predictor here
        ...

    predictor = KairosPredictor(predict_kairos_cloud)
    router = DecisionTreeRouter()
    engine = BacktestEngine(predictor=predictor)

    results = engine.run(df, router, lookback=200)
    print(results)
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple, Any
from enum import Enum
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# ENUMS
# =============================================================================

class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class Regime(Enum):
    RANGE = "range"
    TREND = "trend"
    UNCERTAIN = "uncertain"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Signal:
    direction: Direction
    size: float
    entry: float
    stop: float
    target: float
    strategy_name: str
    confidence: float
    expected_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: Direction
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    strategy_name: str
    exit_reason: str


# =============================================================================
# DISTRIBUTION
# =============================================================================

class KairosDistribution:
    """
    Wraps 60 prediction samples into a statistical distribution.
    Each sample is a single-row DataFrame with columns:
    open, high, low, close, volume, amount.
    """

    def __init__(self, predictions: List[pd.DataFrame]):
        self.predictions = predictions
        self.df = pd.concat(predictions, ignore_index=True)
        self.stats: Dict[str, Dict[str, float]] = {}
        self._compute_stats()

    def _compute_stats(self):
        for col in ["open", "high", "low", "close"]:
            if col not in self.df.columns:
                continue
            arr = self.df[col].values.astype(float)
            self.stats[col] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "skew": float(stats.skew(arr)),
                "kurt": float(stats.kurtosis(arr)),
                "median": float(np.median(arr)),
                "pct_5": float(np.percentile(arr, 5)),
                "pct_10": float(np.percentile(arr, 10)),
                "pct_15": float(np.percentile(arr, 15)),
                "pct_20": float(np.percentile(arr, 20)),
                "pct_25": float(np.percentile(arr, 25)),
                "pct_50": float(np.percentile(arr, 50)),
                "pct_75": float(np.percentile(arr, 75)),
                "pct_80": float(np.percentile(arr, 80)),
                "pct_85": float(np.percentile(arr, 85)),
                "pct_90": float(np.percentile(arr, 90)),
                "pct_95": float(np.percentile(arr, 95)),
            }

    def entropy(self, col: str = "close", bins: int = 20) -> float:
        arr = self.df[col].values.astype(float)
        hist, _ = np.histogram(arr, bins=bins)
        hist = hist[hist > 0].astype(float)
        if len(hist) == 0:
            return 0.0
        pmf = hist / hist.sum()
        return float(-np.sum(pmf * np.log(pmf)))

    def cdf(self, price: float, col: str = "close") -> float:
        return float(stats.percentileofscore(self.df[col].values, price) / 100.0)

    def pdf(self, price: float, col: str = "close") -> float:
        kde = stats.gaussian_kde(self.df[col].values.astype(float))
        return float(kde(price)[0])

    def predicted_sharpe(self, col: str = "close") -> float:
        s = self.stats.get(col, {})
        if s.get("std", 0) > 0:
            return s["mean"] / s["std"]
        return 0.0

    def kelly_fraction(self, entry: float, target: float, stop: float,
                       col: str = "close") -> float:
        values = self.df[col].values.astype(float)
        p_win = float(np.mean(values >= target))
        p_loss = float(np.mean(values <= stop))
        if p_loss == 0:
            return 1.0
        if entry == stop:
            return 0.0
        b = (target - entry) / (entry - stop)
        if b <= 0:
            return 0.0
        f = (p_win * b - p_loss) / b
        return float(max(0.0, min(f, 1.0)))

    def expected_value(self, entry: float, target: float, stop: float,
                       col: str = "close") -> float:
        values = self.df[col].values.astype(float)
        p_win = float(np.mean(values >= target))
        p_loss = float(np.mean(values <= stop))
        p_neutral = max(0.0, 1.0 - p_win - p_loss)
        win_r = target - entry
        loss_r = entry - stop
        return float(p_win * win_r + p_loss * -loss_r + p_neutral * 0.0)

    def overlap_coefficient(self, other: "KairosDistribution",
                            col: str = "close") -> float:
        a = self.df[col].values.astype(float)
        b = other.df[col].values.astype(float)
        min_val = min(a.min(), b.min())
        max_val = max(a.max(), b.max())
        x = np.linspace(min_val, max_val, 1000)
        kde_a = stats.gaussian_kde(a)
        kde_b = stats.gaussian_kde(b)
        return float(np.sum(np.minimum(kde_a(x), kde_b(x))) * (x[1] - x[0]))

    def is_bimodal(self, col: str = "close") -> bool:
        arr = self.df[col].values.astype(float)
        if len(arr) < 10:
            return False
        kde = stats.gaussian_kde(arr)
        x = np.linspace(arr.min(), arr.max(), 1000)
        y = kde(x)
        peaks = 0
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                peaks += 1
        return peaks >= 2

    def coefficient_of_variation(self, col: str = "close") -> float:
        s = self.stats.get(col, {})
        if s.get("mean", 0) != 0:
            return s["std"] / abs(s["mean"])
        return 1.0

    def predicted_range(self, col: str = "close") -> float:
        s = self.stats.get(col, {})
        return s.get("pct_90", 0) - s.get("pct_10", 0)


# =============================================================================
# PREDICTOR WRAPPER
# =============================================================================

class KairosPredictor:
    """
    Thin wrapper around the user's predict_kairos_cloud function.
    """

    def __init__(self, predict_fn: Callable[[pd.DataFrame], List[pd.DataFrame]]):
        self.predict_fn = predict_fn

    def predict(self, history: pd.DataFrame) -> KairosDistribution:
        predictions = self.predict_fn(history)
        return KairosDistribution(predictions)


# =============================================================================
# STRATEGY BASE
# =============================================================================

class Strategy:
    """Base class for all strategies."""
    name: str = "base"

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict) -> Optional[Signal]:
        raise NotImplementedError


# =============================================================================
# STRATEGIES 1-6: CORE DISTRIBUTION STRATEGIES
# =============================================================================

class PercentileEntryStrategy(Strategy):
    """
    Long if current price is at or below the 15th percentile of predicted close.
    Short if at or above the 85th percentile.
    """
    name = "percentile_entry"

    def __init__(self, long_pct: float = 15.0, short_pct: float = 85.0,
                 stop_pct: float = 10.0, target_pct: float = 85.0):
        self.long_pct = long_pct / 100.0
        self.short_pct = short_pct / 100.0
        self.stop_pct = int(stop_pct)
        self.target_pct = int(target_pct)

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        cdf = dist.cdf(current_price)

        if cdf <= self.long_pct:
            direction = Direction.LONG
            stop = s[f"pct_{self.stop_pct}"]
            target = s[f"pct_{self.target_pct}"]
        elif cdf >= self.short_pct:
            direction = Direction.SHORT
            stop = s[f"pct_{self.target_pct}"]
            target = s[f"pct_{self.stop_pct}"]
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        size = min(kelly * 0.5, 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=1.0 - abs(cdf - 0.5) * 2.0,
            expected_value=ev,
        )


class DynamicBracketStrategy(Strategy):
    """
    Uses 10th / 90th percentiles as hard stops and targets.
    Sizes by inverse variance.
    """
    name = "dynamic_bracket"

    def __init__(self, stop_pct: float = 10.0, target_pct: float = 90.0):
        self.stop_pct = int(stop_pct)
        self.target_pct = int(target_pct)

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        mean = s["mean"]

        if mean > current_price:
            direction = Direction.LONG
            stop = s[f"pct_{self.stop_pct}"]
            target = s[f"pct_{self.target_pct}"]
        elif mean < current_price:
            direction = Direction.SHORT
            stop = s[f"pct_{self.target_pct}"]
            target = s[f"pct_{self.stop_pct}"]
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        var = s["std"] ** 2
        size = 0.1 / (var + 0.001)
        size = min(size, 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(1.0, 1.0 / (s["std"] / current_price + 0.001)),
            expected_value=ev,
        )


class SkewStrategy(Strategy):
    """
    Trade in the direction of skew.
    Right skew -> long. Left skew -> short.
    """
    name = "skew"

    def __init__(self, skew_threshold: float = 0.3):
        self.skew_threshold = skew_threshold

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        skew = s["skew"]

        if skew > self.skew_threshold:
            direction = Direction.LONG
        elif skew < -self.skew_threshold:
            direction = Direction.SHORT
        else:
            return None

        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(skew),
            expected_value=ev,
        )


class RangeTradingStrategy(Strategy):
    """
    Mean reversion inside predicted range.
    Buy near predicted low, sell near predicted high.
    """
    name = "range_trading"

    def __init__(self, range_threshold: float = 0.03):
        self.range_threshold = range_threshold

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price

        if pred_range > self.range_threshold:
            return None

        low_dist = dist.stats["low"]
        high_dist = dist.stats["high"]

        if current_price <= low_dist["pct_50"] * 1.005:
            direction = Direction.LONG
            stop = low_dist["pct_10"]
            target = high_dist["pct_50"]
        elif current_price >= high_dist["pct_50"] * 0.995:
            direction = Direction.SHORT
            stop = high_dist["pct_90"]
            target = low_dist["pct_50"]
        else:
            return None

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
            confidence=1.0 - pred_range / self.range_threshold,
            expected_value=ev,
        )


class TrendFollowingStrategy(Strategy):
    """
    If predicted close is far from current and distribution is tight.
    """
    name = "trend_following"

    def __init__(self, min_move_pct: float = 0.01, max_volatility_pct: float = 0.03):
        self.min_move_pct = min_move_pct
        self.max_volatility_pct = max_volatility_pct

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        mean = s["mean"]
        move = (mean - current_price) / current_price
        vol = s["std"] / current_price

        if abs(move) < self.min_move_pct or vol > self.max_volatility_pct:
            return None

        direction = Direction.LONG if move > 0 else Direction.SHORT
        stop = s["pct_25"] if direction == Direction.LONG else s["pct_75"]
        target = s["pct_95"] if direction == Direction.LONG else s["pct_5"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(move) / vol if vol > 0 else 0.0,
            expected_value=ev,
        )


class VolatilityArbStrategy(Strategy):
    """
    Compare predicted realized vol to implied vol.
    Requires 'implied_vol' in context.
    """
    name = "volatility_arb"

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def generate_signal(self, dist, current_price, history, context):
        iv = context.get("implied_vol")
        if iv is None:
            return None

        s = dist.stats["close"]
        pred_vol = s["std"] / current_price

        if pred_vol < iv * (1.0 - self.threshold):
            return Signal(
                direction=Direction.FLAT, size=0.0, entry=current_price,
                stop=0.0, target=0.0, strategy_name=self.name,
                confidence=0.0, expected_value=0.0,
                metadata={"action": "sell_straddle", "pred_vol": pred_vol, "iv": iv}
            )
        elif pred_vol > iv * (1.0 + self.threshold):
            return Signal(
                direction=Direction.FLAT, size=0.0, entry=current_price,
                stop=0.0, target=0.0, strategy_name=self.name,
                confidence=0.0, expected_value=0.0,
                metadata={"action": "buy_straddle", "pred_vol": pred_vol, "iv": iv}
            )
        return None


# =============================================================================
# STRATEGIES 7-12: HIGH/LOW & OPEN-GAP STRATEGIES
# =============================================================================

class HighLowStrategy(Strategy):
    """
    Specifically trades predicted high and low (the strongest Kairos signal).
    If predicted high is only slightly above current, short with target at predicted low.
    If predicted low is only slightly below current, long with target at predicted high.
    """
    name = "high_low"

    def __init__(self, proximity_pct: float = 0.005):
        self.proximity_pct = proximity_pct

    def generate_signal(self, dist, current_price, history, context):
        h = dist.stats["high"]["mean"]
        l = dist.stats["low"]["mean"]
        c = dist.stats["close"]["mean"]

        if (h - current_price) / current_price < self.proximity_pct:
            direction = Direction.SHORT
            stop = h * 1.01
            target = l
        elif (current_price - l) / current_price < self.proximity_pct:
            direction = Direction.LONG
            stop = l * 0.99
            target = h
        else:
            return None

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
            confidence=1.0 - abs((c - current_price) / current_price) / self.proximity_pct,
            expected_value=ev,
        )


class OpenGapStrategy(Strategy):
    """
    Trade the gap between predicted open and current close.
    If predicted open is significantly higher, long at close for gap fill or continuation.
    """
    name = "open_gap"

    def __init__(self, gap_threshold: float = 0.005):
        self.gap_threshold = gap_threshold

    def generate_signal(self, dist, current_price, history, context):
        pred_open = dist.stats["open"]["mean"]
        gap = (pred_open - current_price) / current_price

        if abs(gap) < self.gap_threshold:
            return None

        direction = Direction.LONG if gap > 0 else Direction.SHORT
        s = dist.stats["close"]
        stop = s["pct_25"] if direction == Direction.LONG else s["pct_75"]
        target = s["pct_75"] if direction == Direction.LONG else s["pct_25"]

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
            confidence=abs(gap) / self.gap_threshold,
            expected_value=ev,
        )


class FadeExtremeStrategy(Strategy):
    """
    Fade moves when predicted range is tight and price is near an extreme.
    """
    name = "fade_extreme"

    def __init__(self, range_threshold: float = 0.02, extreme_pct: float = 90.0):
        self.range_threshold = range_threshold
        self.extreme_pct = extreme_pct / 100.0

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price
        if pred_range > self.range_threshold:
            return None

        cdf = dist.cdf(current_price)
        if cdf >= self.extreme_pct:
            direction = Direction.SHORT
            stop = s["pct_95"]
            target = s["pct_50"]
        elif cdf <= (1.0 - self.extreme_pct):
            direction = Direction.LONG
            stop = s["pct_5"]
            target = s["pct_50"]
        else:
            return None

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
            confidence=abs(cdf - 0.5) * 2.0,
            expected_value=ev,
        )


class MomentumContinuationStrategy(Strategy):
    """
    When predicted close is extreme and distribution is tight (high conviction breakout).
    """
    name = "momentum_continuation"

    def __init__(self, move_threshold: float = 0.015, max_cv: float = 0.02):
        self.move_threshold = move_threshold
        self.max_cv = max_cv

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        move = (s["mean"] - current_price) / current_price
        cv = dist.coefficient_of_variation()

        if abs(move) < self.move_threshold or cv > self.max_cv:
            return None

        direction = Direction.LONG if move > 0 else Direction.SHORT
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_95"] if direction == Direction.LONG else s["pct_5"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(move) / cv if cv > 0 else 0.0,
            expected_value=ev,
        )


class ExpectedValueStrategy(Strategy):
    """
    Pure expected-value maximization. No directional bias beyond the math.
    Evaluates long and short EV and picks the better side.
    """
    name = "expected_value"

    def __init__(self, target_atr_mult: float = 1.5, stop_atr_mult: float = 1.0):
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult

    def _atr(self, history: pd.DataFrame) -> float:
        high = history["high"].values
        low = history["low"].values
        close = history["close"].values
        tr1 = high[-1] - low[-1]
        tr2 = abs(high[-1] - close[-2]) if len(close) > 1 else tr1
        tr3 = abs(low[-1] - close[-2]) if len(close) > 1 else tr1
        return float(np.mean([tr1, tr2, tr3]))

    def generate_signal(self, dist, current_price, history, context):
        atr = self._atr(history)
        if atr == 0:
            return None

        s = dist.stats["close"]
        mean = s["mean"]

        # Long setup
        l_target = current_price + atr * self.target_atr_mult
        l_stop = current_price - atr * self.stop_atr_mult
        l_ev = dist.expected_value(current_price, l_target, l_stop)

        # Short setup
        s_target = current_price - atr * self.target_atr_mult
        s_stop = current_price + atr * self.stop_atr_mult
        s_ev = dist.expected_value(current_price, s_target, s_stop)

        if l_ev <= 0 and s_ev <= 0:
            return None

        if l_ev > s_ev:
            direction = Direction.LONG
            target = l_target
            stop = l_stop
            ev = l_ev
        else:
            direction = Direction.SHORT
            target = s_target
            stop = s_stop
            ev = s_ev

        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(ev) / (atr + 0.001),
            expected_value=ev,
        )


class MartingaleFloorStrategy(Strategy):
    """
    Scale in near predicted low with a hard statistical floor.
    Only enters if current price is within 1% of predicted low and
    predicted low std is tight (high confidence floor).
    """
    name = "martingale_floor"

    def __init__(self, proximity_pct: float = 0.01, max_floor_std_pct: float = 0.005):
        self.proximity_pct = proximity_pct
        self.max_floor_std_pct = max_floor_std_pct

    def generate_signal(self, dist, current_price, history, context):
        l_mean = dist.stats["low"]["mean"]
        l_std = dist.stats["low"]["std"]

        if (current_price - l_mean) / current_price > self.proximity_pct:
            return None
        if l_std / current_price > self.max_floor_std_pct:
            return None

        direction = Direction.LONG
        stop = l_mean * 0.98
        target = dist.stats["high"]["mean"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=0.8,  # Larger size due to tight floor
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=1.0 - (l_std / current_price) / self.max_floor_std_pct,
            expected_value=ev,
        )


# =============================================================================
# STRATEGIES 13-16: TECHNICAL INDICATOR + FOREKNOWLEDGE
# =============================================================================

class RSIFilterStrategy(Strategy):
    """
    Only take directional signals when RSI confirms.
    """
    name = "rsi_filter"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _rsi(self, history: pd.DataFrame) -> float:
        close = history["close"].values[-self.period:]
        if len(close) < 2:
            return 50.0
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signal(self, dist, current_price, history, context):
        rsi = self._rsi(history)
        s = dist.stats["close"]
        mean = s["mean"]

        if mean > current_price and rsi < self.oversold:
            direction = Direction.LONG
        elif mean < current_price and rsi > self.overbought:
            direction = Direction.SHORT
        else:
            return None

        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(50.0 - rsi) / 50.0,
            expected_value=ev,
        )


class MACDFilterStrategy(Strategy):
    """
    Only take signals when MACD crossover aligns with Kairos prediction.
    """
    name = "macd_filter"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def _macd(self, history: pd.DataFrame) -> Tuple[float, float, float]:
        close = history["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=self.slow, adjust=False).mean().iloc[-1]
        macd_line = ema_fast - ema_slow
        signal_line = close.ewm(span=self.signal, adjust=False).mean().iloc[-1]
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_signal(self, dist, current_price, history, context):
        macd_line, signal_line, hist = self._macd(history)
        s = dist.stats["close"]
        mean = s["mean"]

        if mean > current_price and macd_line > signal_line:
            direction = Direction.LONG
        elif mean < current_price and macd_line < signal_line:
            direction = Direction.SHORT
        else:
            return None

        stop = s["pct_15"] if direction == Direction.LONG else s["pct_85"]
        target = s["pct_85"] if direction == Direction.LONG else s["pct_15"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(hist) / (abs(macd_line) + 0.001),
            expected_value=ev,
        )


class BollingerValidationStrategy(Strategy):
    """
    Compare predicted range to Bollinger Band width.
    If predicted high is below upper BB, the breakout is fake.
    """
    name = "bollinger_validation"

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def _bb(self, history: pd.DataFrame) -> Tuple[float, float, float]:
        close = history["close"].values[-self.period:]
        if len(close) < self.period:
            return current_price, current_price, current_price
        mid = np.mean(close)
        std = np.std(close)
        return mid - self.std_dev * std, mid, mid + self.std_dev * std

    def generate_signal(self, dist, current_price, history, context):
        lower, mid, upper = self._bb(history)
        s = dist.stats["close"]
        pred_high = dist.stats["high"]["mean"]
        pred_low = dist.stats["low"]["mean"]

        if current_price >= upper and pred_high < upper:
            direction = Direction.SHORT
            stop = s["pct_90"]
            target = s["pct_25"]
        elif current_price <= lower and pred_low > lower:
            direction = Direction.LONG
            stop = s["pct_10"]
            target = s["pct_75"]
        else:
            return None

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
            confidence=0.7,
            expected_value=ev,
        )


class SupportConfluenceStrategy(Strategy):
    """
    Check if predicted low aligns with a recent volume-weighted support level.
    """
    name = "support_confluence"

    def __init__(self, lookback: int = 20, tolerance_pct: float = 0.01):
        self.lookback = lookback
        self.tolerance_pct = tolerance_pct

    def _support(self, history: pd.DataFrame) -> Optional[float]:
        sub = history.iloc[-self.lookback:]
        if len(sub) < 5:
            return None
        # Volume-weighted low as support proxy
        vol = sub["volume"].values
        low = sub["low"].values
        if vol.sum() == 0:
            return float(np.min(low))
        return float(np.average(low, weights=vol))

    def generate_signal(self, dist, current_price, history, context):
        support = self._support(history)
        if support is None:
            return None

        pred_low = dist.stats["low"]["mean"]
        alignment = abs(pred_low - support) / support

        if alignment > self.tolerance_pct:
            return None

        if current_price <= pred_low * 1.01:
            direction = Direction.LONG
            stop = support * 0.98
            target = dist.stats["high"]["mean"]
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=0.6,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=1.0 - alignment / self.tolerance_pct,
            expected_value=ev,
        )


# =============================================================================
# STRATEGIES 17-18: META / COMPOSITE
# =============================================================================

class InverseVarianceSizingStrategy(Strategy):
    """
    Pure inverse-variance sizing on predicted direction.
    """
    name = "inverse_variance"

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        mean = s["mean"]
        if mean == current_price:
            return None

        direction = Direction.LONG if mean > current_price else Direction.SHORT
        var = s["std"] ** 2
        if var == 0:
            return None

        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(0.2 / var, 1.0)
        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=1.0 / (1.0 + var * 100),
            expected_value=ev,
        )


class CloseDirectionStrategy(Strategy):
    """
    The simplest strategy: predicted close > current -> long, else short.
    Sized by predicted Sharpe.
    """
    name = "close_direction"

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        mean = s["mean"]
        if mean > current_price:
            direction = Direction.LONG
        elif mean < current_price:
            direction = Direction.SHORT
        else:
            return None

        stop = s["pct_20"] if direction == Direction.LONG else s["pct_80"]
        target = s["pct_80"] if direction == Direction.LONG else s["pct_20"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        sharpe = dist.predicted_sharpe()
        size = min(abs(sharpe) * 0.5, 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(sharpe), 1.0),
            expected_value=ev,
        )


# =============================================================================
# DECISION TREE ROUTER
# =============================================================================

class DecisionTreeRouter:
    """
    Routes to strategies based on distribution properties.
    """

    def __init__(self,
                 entropy_threshold: float = 3.0,
                 range_threshold: float = 0.03,
                 ev_threshold: float = 0.0,
                 custom_strategies: Optional[Dict[str, List[Strategy]]] = None):
        self.entropy_threshold = entropy_threshold
        self.range_threshold = range_threshold
        self.ev_threshold = ev_threshold

        if custom_strategies:
            self.strategies = custom_strategies
        else:
            self.strategies = {
                "range": [
                    RangeTradingStrategy(),
                    PercentileEntryStrategy(),
                    FadeExtremeStrategy(),
                    BollingerValidationStrategy(),
                    HighLowStrategy(),
                    MartingaleFloorStrategy(),
                ],
                "trend": [
                    TrendFollowingStrategy(),
                    SkewStrategy(),
                    MomentumContinuationStrategy(),
                    MACDFilterStrategy(),
                    RSIFilterStrategy(),
                    DynamicBracketStrategy(),
                    CloseDirectionStrategy(),
                ],
                "uncertain": [
                    ExpectedValueStrategy(),
                    InverseVarianceSizingStrategy(),
                ],
            }

        self.meta_filters = {
            "entropy": True,
            "bimodal": True,
            "ev": True,
        }

    def detect_regime(self, dist: KairosDistribution, current_price: float) -> Regime:
        s = dist.stats["close"]
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price
        cv = dist.coefficient_of_variation()

        if pred_range < self.range_threshold and cv < 0.02:
            return Regime.RANGE
        elif pred_range > self.range_threshold * 1.5 or cv > 0.04:
            return Regime.TREND
        return Regime.UNCERTAIN

    def route(self, dist: KairosDistribution, current_price: float,
              history: pd.DataFrame, context: Dict) -> Optional[Signal]:
        # Filter 1: Entropy
        if self.meta_filters["entropy"]:
            ent = dist.entropy()
            if ent > self.entropy_threshold:
                return None

        # Filter 2: Bimodality
        if self.meta_filters["bimodal"] and dist.is_bimodal():
            return None

        # Filter 3: Regime
        regime = self.detect_regime(dist, current_price)
        candidates = self.strategies.get(regime.value, [])

        best_signal = None
        best_score = -float("inf")

        for strat in candidates:
            signal = strat.generate_signal(dist, current_price, history, context)
            if signal is None:
                continue

            # Filter 4: EV
            if self.meta_filters["ev"] and signal.expected_value <= self.ev_threshold:
                continue

            score = signal.expected_value * signal.confidence * signal.size
            if score > best_score:
                best_score = score
                best_signal = signal

        return best_signal


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Walk-forward backtester.
    Assumes predictions are made at close of day i for day i+1.
    Entry at day i+1 open, exit based on day i+1 high/low/close.
    """

    def __init__(self,
                 predictor: KairosPredictor,
                 fee_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 initial_capital: float = 10000.0):
        self.predictor = predictor
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.initial_capital = initial_capital

        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.signals: List[Tuple[pd.Timestamp, Signal]] = []
        self.daily_pnl: List[Tuple[pd.Timestamp, float]] = []

    def run(self, df: pd.DataFrame, router: DecisionTreeRouter,
            lookback: int = 100) -> Dict:
        capital = self.initial_capital
        position = None
        current_trade_meta = None
        prev_capital = capital

        for i in range(lookback, len(df) - 1):
            today = df.iloc[i]
            tomorrow = df.iloc[i + 1]
            date = df.index[i]
            current_price = float(today["close"])

            history = df.iloc[:i + 1]
            dist = self.predictor.predict(history)

            context = {
                "date": date,
                "current_price": current_price,
                "capital": capital,
            }
            signal = router.route(dist, current_price, history, context)

            if signal:
                self.signals.append((date, signal))

            # Manage existing position
            if position is not None:
                exit_price, exit_reason = self._check_exit(
                    position, tomorrow
                )
                if exit_price is not None:
                    pnl = self._calculate_pnl(position, exit_price)
                    capital += pnl
                    self.trades.append(Trade(
                        entry_date=current_trade_meta["entry_date"],
                        exit_date=df.index[i + 1],
                        direction=position["direction"],
                        entry_price=current_trade_meta["entry_price"],
                        exit_price=exit_price,
                        size=position["size"],
                        pnl=pnl,
                        pnl_pct=pnl / (current_trade_meta["entry_price"] * position["size"]),
                        strategy_name=current_trade_meta["strategy_name"],
                        exit_reason=exit_reason,
                    ))
                    position = None
                    current_trade_meta = None

            # Enter new position
            if position is None and signal and signal.direction != Direction.FLAT:
                entry_price = float(tomorrow["open"]) * (
                    1.0 + self.slippage_pct * signal.direction.value
                )
                notional = signal.size * capital
                fee = notional * self.fee_pct
                capital -= fee

                position = {
                    "direction": signal.direction,
                    "size": notional / entry_price,
                    "entry": entry_price,
                    "stop": signal.stop,
                    "target": signal.target,
                }
                current_trade_meta = {
                    "entry_date": df.index[i + 1],
                    "entry_price": entry_price,
                    "strategy_name": signal.strategy_name,
                }

            self.equity_curve.append((date, capital))
            self.daily_pnl.append((date, capital - prev_capital))
            prev_capital = capital

        return self._compute_metrics()

    def _check_exit(self, position: Dict, tomorrow: pd.Series) -> Tuple[Optional[float], Optional[str]]:
        direction = position["direction"]
        stop = position["stop"]
        target = position["target"]
        open_price = float(tomorrow["open"])
        high = float(tomorrow["high"])
        low = float(tomorrow["low"])
        close = float(tomorrow["close"])

        if direction == Direction.LONG:
            if open_price <= stop:
                return open_price, "stop_open"
            if open_price >= target:
                return open_price, "target_open"
            if low <= stop:
                return stop, "stop"
            if high >= target:
                return target, "target"
            return close, "close"
        else:
            if open_price >= stop:
                return open_price, "stop_open"
            if open_price <= target:
                return open_price, "target_open"
            if high >= stop:
                return stop, "stop"
            if low <= target:
                return target, "target"
            return close, "close"

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        if position["direction"] == Direction.LONG:
            gross = (exit_price - position["entry"]) * position["size"]
        else:
            gross = (position["entry"] - exit_price) * position["size"]
        fee = exit_price * position["size"] * self.fee_pct
        return gross - fee

    def _compute_metrics(self) -> Dict:
        if not self.trades:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "num_trades": 0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity = [e for _, e in self.equity_curve]
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        returns = np.diff(equity) / np.array(equity[:-1])
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        profit_factor = float(abs(sum(wins) / sum(losses))) if sum(losses) != 0 else float("inf")

        return {
            "total_return": float((equity[-1] - self.initial_capital) / self.initial_capital),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": float(len(wins) / len(pnls)) if pnls else 0.0,
            "profit_factor": profit_factor,
            "num_trades": len(self.trades),
            "avg_trade": float(np.mean(pnls)) if pnls else 0.0,
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
        }

    def run_strategy_comparison(self, df: pd.DataFrame, strategies: List[Strategy],
                                lookback: int = 100) -> pd.DataFrame:
        """
        Run each strategy in isolation and return a comparison DataFrame.
        """
        results = []
        for strat in strategies:
            router = DecisionTreeRouter(
                entropy_threshold=999.0,  # Disable meta filters
                custom_strategies={
                    "range": [strat],
                    "trend": [strat],
                    "uncertain": [strat],
                }
            )
            self.trades = []
            self.equity_curve = []
            self.signals = []
            self.daily_pnl = []
            metrics = self.run(df, router, lookback)
            metrics["strategy"] = strat.name
            results.append(metrics)
        return pd.DataFrame(results)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Replace this stub with your actual predictor.
    def predict_kairos_cloud(df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Your 60-run predictor.
        Returns a list of 60 single-row DataFrames with columns:
        open, high, low, close, volume, amount
        Index should be the predicted date.
        """
        raise NotImplementedError("Replace with your predict_kairos_cloud implementation")

    # Example data loading (uncomment and adapt):
    # df = pd.read_csv("btc_daily.csv", index_col=0, parse_dates=True)
    # df = df[["open", "high", "low", "close", "volume", "amount"]]

    # predictor = KairosPredictor(predict_kairos_cloud)
    # router = DecisionTreeRouter()
    # engine = BacktestEngine(predictor=predictor, fee_pct=0.001, slippage_pct=0.0005)
    # results = engine.run(df, router, lookback=200)
    # print(results)

    # Strategy comparison:
    # all_strategies = [
    #     PercentileEntryStrategy(),
    #     DynamicBracketStrategy(),
    #     SkewStrategy(),
    #     RangeTradingStrategy(),
    #     TrendFollowingStrategy(),
    #     HighLowStrategy(),
    #     OpenGapStrategy(),
    #     FadeExtremeStrategy(),
    #     MomentumContinuationStrategy(),
    #     ExpectedValueStrategy(),
    #     MartingaleFloorStrategy(),
    #     RSIFilterStrategy(),
    #     MACDFilterStrategy(),
    #     BollingerValidationStrategy(),
    #     SupportConfluenceStrategy(),
    #     InverseVarianceSizingStrategy(),
    #     CloseDirectionStrategy(),
    # ]
    # comparison = engine.run_strategy_comparison(df, all_strategies, lookback=200)
    # print(comparison)
    pass
