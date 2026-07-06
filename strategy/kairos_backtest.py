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
# GLOBAL SETTINGS
# =============================================================================

class KairosSettings:
    """Central store for CLI-configured runtime settings.

    Call KairosSettings.configure(args) after argparse.parse_args(), then
    read KairosSettings.<attr> from any strategy module.
    """
    symbol: str = "BTC-USD"
    lookback: int = 300
    pred_len: int = 60
    pred_samples: int = 100
    initial_capital: float = 100_000.0
    output_dir: str = "./output"
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    no_prediction: bool = False
    interval: str = "1d"
    assets: list = None   # None → caller falls back to default asset list
    backtest_period: str = "6m"

    @classmethod
    def configure(cls, args) -> None:
        for attr in ("symbol", "lookback", "pred_len", "pred_samples",
                     "initial_capital", "output_dir", "model", "tokenizer",
                     "no_prediction", "interval", "assets", "backtest_period"):
            if hasattr(args, attr) and getattr(args, attr) is not None:
                setattr(cls, attr, getattr(args, attr))


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

def fast_concat(predictions: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Fast equivalent of pd.concat(predictions, ignore_index=True) for the
    common case of many small frames with identical columns and dtypes.
    Falls back to pd.concat otherwise (identical output either way).
    """
    if predictions:
        first = predictions[0]
        cols = first.columns
        col_list = cols.tolist()
        arrs = []
        ok = True
        for p in predictions:
            pc = p.columns
            if pc is not cols and pc.tolist() != col_list:
                ok = False
                break
            arrs.append(p.to_numpy())
        if ok and arrs:
            dt = arrs[0].dtype
            if dt != object and all(a.dtype == dt for a in arrs) \
                    and (first.dtypes == dt).all():
                return pd.DataFrame(np.vstack(arrs), columns=cols.copy())
    return pd.concat(predictions, ignore_index=True)


_dist_memo: dict = {}


def distribution_for(predictions: List[pd.DataFrame]) -> "KairosDistribution":
    """
    Memoized KairosDistribution construction, keyed by the identity of the
    predictions list. The list is retained in the memo so ids stay valid.
    Callers that reuse the same cached predictions list get the same
    (immutable in practice) distribution back instead of rebuilding it.
    """
    key = id(predictions)
    entry = _dist_memo.get(key)
    if entry is not None and entry[0] is predictions:
        return entry[1]
    dist = KairosDistribution(predictions)
    _dist_memo[key] = (predictions, dist)
    return dist


class KairosDistribution:
    """
    Wraps 60 prediction samples into a statistical distribution.
    Each sample is a single-row DataFrame with columns:
    open, high, low, close, volume, amount.
    """

    def __init__(self, predictions: List[pd.DataFrame]):
        self.predictions = predictions
        self.df = fast_concat(predictions)
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
                "pct_16": float(np.percentile(arr, 16)),
                "pct_20": float(np.percentile(arr, 20)),
                "pct_25": float(np.percentile(arr, 25)),
                "pct_50": float(np.percentile(arr, 50)),
                "pct_75": float(np.percentile(arr, 75)),
                "pct_80": float(np.percentile(arr, 80)),
                "pct_84": float(np.percentile(arr, 84)),
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

    @classmethod
    def from_bar(cls, bar: pd.Series, n_samples: int = 100,
                 center: float = None) -> "KairosDistribution":
        """Build a realized distribution from a single OHLCV bar.

        Generates N samples representing possible intraday paths bounded by the
        bar's actual high/low.  Closes follow a truncated normal; by default
        centred on the actual close.  Pass center=current_price (the previous
        bar's close) to anchor stop/target relative to the trade entry rather
        than the oracle close - this keeps pct_20 below entry for LONG signals
        and pct_80 above entry for SHORT signals, producing proper risk/reward.
        """
        actual_open = float(bar["open"])
        actual_high = float(bar["high"])
        actual_low = float(bar["low"])
        actual_close = float(bar["close"])
        actual_volume = float(bar.get("volume", 1.0))

        # Guard against zero-range bars
        if actual_high <= actual_low:
            actual_high = max(actual_open, actual_close) * 1.0005
            actual_low = min(actual_open, actual_close) * 0.9995

        seed = int(abs(actual_close) * 1000) % (2 ** 31)
        rng = np.random.default_rng(seed)

        # Closes: truncated normal, centre defaults to actual_close.
        # When center is provided (e.g. the prior bar's close = trade entry),
        # the distribution spans the actual H/L range anchored at entry so
        # that stop/target levels are correctly placed relative to entry price.
        span = actual_high - actual_low
        close_center = center if center is not None else actual_close
        closes = np.clip(rng.normal(close_center, span / 4.0, n_samples),
                         actual_low, actual_high)

        rows = []
        for c in closes:
            c = float(c)
            # high must be ≥ max(open, close), bounded by actual high
            max_oc = max(actual_open, c)
            h = float(rng.uniform(max_oc, actual_high)) if actual_high > max_oc else max_oc
            # low must be ≤ min(open, close), bounded by actual low
            min_oc = min(actual_open, c)
            l = float(rng.uniform(actual_low, min_oc)) if min_oc > actual_low else min_oc
            rows.append(pd.DataFrame({
                "open":   [actual_open],
                "high":   [h],
                "low":    [l],
                "close":  [c],
                "volume": [actual_volume],
            }))
        return cls(rows)


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
# TECHNICAL INDICATOR HELPERS
# =============================================================================

def compute_adx(history: pd.DataFrame, n: int = 14) -> float:
    """
    Compute ADX (Average Directional Index) from history.

    ADX measures trend strength from 0-100. Values:
    - 0-25: weak trend
    - 25-50: moderate trend
    - 50-75: strong trend
    - 75-100: very strong trend

    Args:
        history: DataFrame with OHLC data (high, low, close columns required)
        n: period for ADX calculation (default 14)

    Returns:
        ADX value (0-100), or 50.0 (neutral) if insufficient data
    """
    if len(history) < n + 1:
        return 50.0  # Default neutral

    high = history["high"].values
    low = history["low"].values
    close = history["close"].values

    # Compute True Range (TR)
    tr_values = []
    for i in range(1, len(close)):
        h_l = high[i] - low[i]
        h_c = abs(high[i] - close[i - 1])
        l_c = abs(low[i] - close[i - 1])
        tr = max(h_l, h_c, l_c)
        tr_values.append(tr)

    if len(tr_values) == 0:
        return 50.0

    # Compute Directional Movements
    plus_dm_values = []
    minus_dm_values = []
    for i in range(1, len(high)):
        plus_dm = high[i] - high[i - 1]
        minus_dm = low[i - 1] - low[i]

        # Only count if positive and greater than the opposite
        if plus_dm > 0 and plus_dm > minus_dm:
            plus_dm_values.append(plus_dm)
        else:
            plus_dm_values.append(0.0)

        if minus_dm > 0 and minus_dm > plus_dm:
            minus_dm_values.append(minus_dm)
        else:
            minus_dm_values.append(0.0)

    # Compute DI values using average
    if len(tr_values) >= n:
        # Use SMA for numerical stability
        tr_sum = np.mean(tr_values[-n:])
        plus_dm_sum = np.mean(plus_dm_values[-n:])
        minus_dm_sum = np.mean(minus_dm_values[-n:])

        if tr_sum > 0:
            plus_di = 100.0 * plus_dm_sum / tr_sum
            minus_di = 100.0 * minus_dm_sum / tr_sum
        else:
            return 50.0

        # Compute DX
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0.0

        # ADX is smoothed DX (simplified: use current DX as approximation)
        adx = dx
    else:
        adx = 50.0

    return adx


# =============================================================================
# STRATEGY BASE
# =============================================================================

class Strategy:
    """Base class for all strategies."""
    name: str = "base"

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
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


class StochasticFilterStrategy(Strategy):
    """
    Wraps a base strategy with a Stochastic Oscillator + ADX filter.

    Veto LONG signals when %K > overbought UNLESS trend is strong (ADX > adx_trend).
    Veto SHORT signals when %K < oversold UNLESS trend is strong (ADX > adx_trend).
    Otherwise pass base signal through unchanged.

    Uses:
    - %K = 100 * (close - lowest_low) / (highest_high - lowest_low) over k_period bars
    - %D = SMA(%K, d_period)
    - ADX (Average Directional Index) over adx_period bars
    """
    name = "stochastic_filter"

    def __init__(self, base_strategy: Strategy,
                 k_period: int = 14, d_period: int = 3,
                 overbought: float = 80.0, oversold: float = 20.0,
                 adx_period: int = 14, adx_trend: float = 25.0):
        self.base_strategy = base_strategy
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
        self.adx_period = adx_period
        self.adx_trend = adx_trend

    def _compute_stochastic(self, history: pd.DataFrame) -> Tuple[float, float]:
        """
        Compute %K and %D from history.
        Returns: (k_value, d_value)
        """
        if len(history) < self.k_period:
            return 50.0, 50.0

        # Get the last k_period bars
        close = history["close"].values[-self.k_period:]
        high = history["high"].values[-self.k_period:]
        low = history["low"].values[-self.k_period:]

        # Compute %K for the entire history window (need for %D)
        all_close = history["close"].values
        all_high = history["high"].values
        all_low = history["low"].values

        k_values = []
        for i in range(len(all_close) - self.k_period + 1):
            window_high = all_high[i:i + self.k_period].max()
            window_low = all_low[i:i + self.k_period].min()
            range_val = window_high - window_low
            if range_val == 0:
                k_values.append(50.0)
            else:
                k = 100.0 * (all_close[i + self.k_period - 1] - window_low) / range_val
                k_values.append(k)

        if len(k_values) == 0:
            return 50.0, 50.0

        # Current %K
        k_current = k_values[-1]

        # Compute %D as SMA of %K over d_period
        if len(k_values) >= self.d_period:
            d_current = np.mean(k_values[-self.d_period:])
        else:
            d_current = k_current

        return k_current, d_current

    def _compute_adx(self, history: pd.DataFrame) -> float:
        """
        Compute ADX (Average Directional Index) from history.
        Delegates to the module-level compute_adx helper.
        Returns: ADX value (0-100)
        """
        return compute_adx(history, n=self.adx_period)

    def generate_signal(self, dist, current_price, history, context):
        """
        Generate signal from base strategy, then apply stochastic + ADX filter.
        """
        # Get base signal first
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if base_signal is None:
            return None

        # Compute stochastic and ADX
        k_val, d_val = self._compute_stochastic(history)
        adx_val = self._compute_adx(history)

        # Apply veto logic
        if base_signal.direction == Direction.LONG:
            # Veto LONG if %K > overbought UNLESS ADX > adx_trend
            if k_val > self.overbought and adx_val <= self.adx_trend:
                return None
        elif base_signal.direction == Direction.SHORT:
            # Veto SHORT if %K < oversold UNLESS ADX > adx_trend
            if k_val < self.oversold and adx_val <= self.adx_trend:
                return None

        # Pass through base signal unchanged
        return base_signal


class ADXGateStrategy(Strategy):
    """
    Wrapper that gates a base strategy based on trend strength (ADX).

    Routes strategies by market regime:
    - kind="trend": passes base signal only when ADX > trend_min (default 25)
    - kind="reversion": passes base signal only when ADX < reversion_max (default 20)
    - Otherwise returns None, blocking the signal.

    This allows trend-following and mean-reversion strategies to automatically
    adapt to market conditions.

    Example:
        trend_strat = TrendFollowingStrategy()
        gated = ADXGateStrategy(trend_strat, kind="trend", trend_min=25.0)

        reversion_strat = RangeTradingStrategy()
        gated_reversion = ADXGateStrategy(reversion_strat, kind="reversion", reversion_max=20.0)
    """
    name = "adx_gate"

    def __init__(self, base_strategy: Strategy, kind: str,
                 adx_period: int = 14,
                 trend_min: float = 25.0,
                 reversion_max: float = 20.0):
        """
        Initialize ADXGateStrategy.

        Args:
            base_strategy: Strategy to wrap
            kind: "trend" or "reversion" - determines gating behavior
            adx_period: period for ADX calculation (default 14)
            trend_min: ADX threshold for trend-type strategies (default 25.0)
            reversion_max: ADX threshold for reversion-type strategies (default 20.0)

        Raises:
            ValueError: if kind is not "trend" or "reversion"
        """
        if kind not in ("trend", "reversion"):
            raise ValueError(f"kind must be 'trend' or 'reversion', got {kind!r}")

        self.base_strategy = base_strategy
        self.kind = kind
        self.adx_period = adx_period
        self.trend_min = trend_min
        self.reversion_max = reversion_max

    def generate_signal(self, dist, current_price, history, context):
        """
        Generate signal from base strategy, then gate based on ADX.

        Returns:
            Signal from base strategy if gate condition met, None otherwise.
            Always returns Signal or None, never dict.
        """
        # Get base signal first
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if base_signal is None:
            return None

        # Compute ADX
        adx_val = compute_adx(history, n=self.adx_period)

        # Apply gating logic
        if self.kind == "trend":
            # Trend strategies pass only when ADX > trend_min
            if adx_val > self.trend_min:
                return base_signal
            else:
                return None
        else:  # kind == "reversion"
            # Reversion strategies pass only when ADX < reversion_max
            if adx_val < self.reversion_max:
                return base_signal
            else:
                return None


class OBVConfirmationStrategy(Strategy):
    """
    On-Balance Volume (OBV) confirmation filter.

    Computes OBV from realized volume (cumulative volume signed by close-to-close change).
    Calculates the slope of the last slope_window OBV values via linear regression.
    Vetoes signals when OBV slope sign disagrees with signal direction:
    - LONG signals vetoed if slope is negative
    - SHORT signals vetoed if slope is positive
    Flat/zero slope passes through; matching sign passes through unchanged.

    Complements VolumeConfirmationStrategy (which uses predicted volume).
    """
    name = "obv_confirmation"

    def __init__(self, base_strategy: Strategy, slope_window: int = 20):
        """
        Initialize OBVConfirmationStrategy.

        Args:
            base_strategy: Strategy to wrap
            slope_window: Window size for OBV slope calculation (default 20)
        """
        self.base_strategy = base_strategy
        self.slope_window = slope_window

    def _compute_obv(self, history: pd.DataFrame) -> Tuple[List[float], float]:
        """
        Compute OBV values from history.

        OBV is cumulative volume signed by close-to-close change:
        - If close > previous close: add volume
        - If close < previous close: subtract volume
        - If close == previous close: add 0 (no change to OBV)

        Returns:
            (obv_values, current_obv) tuple
        """
        closes = history["close"].values
        volumes = history["volume"].values

        if len(closes) < 2:
            return [0.0], 0.0

        obv_values = [0.0]  # OBV starts at 0
        obv = 0.0

        for i in range(1, len(closes)):
            price_change = closes[i] - closes[i - 1]
            if price_change > 0:
                obv += volumes[i]
            elif price_change < 0:
                obv -= volumes[i]
            # If price_change == 0, add 0 to OBV (no change)
            obv_values.append(obv)

        return obv_values, obv

    def _compute_obv_slope(self, history: pd.DataFrame) -> float:
        """
        Compute the slope of OBV over the last slope_window bars.

        Uses numpy.polyfit with degree 1 (linear regression).
        Returns the slope coefficient.

        Returns:
            slope value (positive = rising OBV, negative = falling OBV)
        """
        obv_values, _ = self._compute_obv(history)

        if len(obv_values) < self.slope_window:
            # Not enough data; return 0 (neutral, no veto)
            return 0.0

        # Get the last slope_window OBV values
        obv_window = obv_values[-self.slope_window:]
        x = np.arange(len(obv_window), dtype=float)
        y = np.array(obv_window, dtype=float)

        # Fit a degree-1 polynomial (linear)
        try:
            coeffs = np.polyfit(x, y, deg=1)
            slope = coeffs[0]  # First element is the slope
        except Exception:
            # If polyfit fails, return 0 (neutral)
            return 0.0

        return slope

    def generate_signal(self, dist, current_price, history, context):
        """
        Generate signal from base strategy, then apply OBV slope filter.

        Returns:
            Signal from base strategy if OBV slope agrees with direction, None if vetoed.
            Always returns Signal or None, never dict.
        """
        # Get base signal first
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if base_signal is None:
            return None

        # Compute OBV slope
        obv_slope = self._compute_obv_slope(history)

        # Apply veto logic
        if base_signal.direction == Direction.LONG:
            # Veto LONG if slope is negative (OBV falling)
            if obv_slope < 0:
                return None
        elif base_signal.direction == Direction.SHORT:
            # Veto SHORT if slope is positive (OBV rising)
            if obv_slope > 0:
                return None

        # Pass through base signal unchanged (flat slope or matching sign)
        return base_signal


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
# STRATEGIES 19-24: NEW WRAPPER & ADVANCED STRATEGIES
# =============================================================================

class VaRPositionCapStrategy(Strategy):
    """
    Wraps a base strategy and caps position size based on Value at Risk (5th percentile).
    Ensures max loss per unit doesn't exceed account risk limit.
    """
    name = "var_position_cap"

    def __init__(self, base_strategy: Strategy, max_account_risk_pct: float = 0.01,
                 capital: float = 1.0):
        self.base_strategy = base_strategy
        self.max_account_risk_pct = max_account_risk_pct
        self.capital = capital

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        var_5 = dist.stats["close"]["pct_5"]
        entry = base_signal.entry

        if base_signal.direction == Direction.LONG:
            max_loss_per_unit = entry - var_5
        elif base_signal.direction == Direction.SHORT:
            max_loss_per_unit = var_5 - entry
        else:
            return base_signal

        if max_loss_per_unit <= 0:
            return base_signal

        account_risk_limit = self.capital * self.max_account_risk_pct
        max_units = account_risk_limit / max_loss_per_unit
        max_notional = max_units * entry
        max_size = max_notional / self.capital

        base_signal.size = min(base_signal.size, max_size)
        return base_signal


class DistributionOverlapStrategy(Strategy):
    """
    Uses overlap coefficient with previous distribution to detect regime.
    High overlap → range trading (mean reversion).
    Low overlap → trend following.
    """
    name = "distribution_overlap"

    def __init__(self, range_threshold: float = 0.85, trend_threshold: float = 0.60):
        self.range_threshold = range_threshold
        self.trend_threshold = trend_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        prev_dist = context.get("prev_dist")
        if prev_dist is None:
            return None

        overlap = dist.overlap_coefficient(prev_dist, col="close")
        s = dist.stats["close"]
        median = s["pct_50"]
        mean = s["mean"]

        if overlap > self.range_threshold:
            # Range-bound: mean reversion toward median
            if current_price < median:
                direction = Direction.LONG
                stop = s["pct_10"]
                target = median
            else:
                direction = Direction.SHORT
                stop = s["pct_90"]
                target = median
        elif overlap < self.trend_threshold:
            # Trend following: follow the mean
            if mean > current_price:
                direction = Direction.LONG
                stop = s["pct_10"]
                target = s["pct_90"]
            elif mean < current_price:
                direction = Direction.SHORT
                stop = s["pct_90"]
                target = s["pct_10"]
            else:
                return None
        else:
            return None

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
            confidence=abs(overlap - 0.5) / 0.5,
            expected_value=ev,
        )


class ModelDecayMonitorStrategy(Strategy):
    """
    Wraps a base strategy and adjusts stops/size based on calibration history.
    Tracks whether realized prices fall within 1-sigma and 2-sigma bands.
    """
    name = "model_decay_monitor"

    def __init__(self, base_strategy: Strategy, lookback: int = 30,
                 target_1sigma: float = 0.68, target_2sigma: float = 0.95,
                 widen_factor: float = 1.5, tighten_factor: float = 0.8):
        self.base_strategy = base_strategy
        self.lookback = lookback
        self.target_1sigma = target_1sigma
        self.target_2sigma = target_2sigma
        self.widen_factor = widen_factor
        self.tighten_factor = tighten_factor
        self.calibration_history = []

    def update_calibration(self, dist: KairosDistribution, realized_close: float) -> None:
        """Record prediction accuracy."""
        mean = dist.stats["close"]["mean"]
        std = dist.stats["close"]["std"]
        in_1s = (mean - std) <= realized_close <= (mean + std)
        in_2s = (mean - 2 * std) <= realized_close <= (mean + 2 * std)
        self.calibration_history.append((mean, std, realized_close, in_1s, in_2s))
        if len(self.calibration_history) > self.lookback * 2:
            self.calibration_history.pop(0)

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        if len(self.calibration_history) < self.lookback:
            return base_signal

        recent = list(self.calibration_history)[-self.lookback:]
        hit_rate_1s = np.mean([entry[3] for entry in recent])

        if hit_rate_1s < self.target_1sigma * 0.8:
            # Predictions too tight, widen stops
            calibration_factor = self.widen_factor
            size_factor = 0.5
        elif hit_rate_1s > self.target_1sigma * 1.2:
            # Predictions too loose, tighten
            calibration_factor = self.tighten_factor
            size_factor = 1.0
        else:
            calibration_factor = 1.0
            size_factor = 1.0

        # Apply calibration to stop
        entry = base_signal.entry
        if base_signal.direction == Direction.LONG:
            stop_distance = entry - base_signal.stop
            base_signal.stop = entry - (stop_distance * calibration_factor)
        elif base_signal.direction == Direction.SHORT:
            stop_distance = base_signal.stop - entry
            base_signal.stop = entry + (stop_distance * calibration_factor)

        base_signal.size = min(base_signal.size * size_factor, 1.0)
        return base_signal


class OvernightExposureFilter(Strategy):
    """
    Wrapper that returns FLAT if next-day predicted range doesn't favor the current position.
    Useful for avoiding hold-through-news trades.
    """
    name = "overnight_filter"

    def __init__(self, base_strategy: Strategy):
        self.base_strategy = base_strategy

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        current_pos = context.get("current_position")

        if current_pos is None:
            # No existing position, pass through
            return self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)

        pred_high = dist.stats.get("high", {}).get("mean", dist.stats["close"]["pct_90"])
        pred_low = dist.stats.get("low", {}).get("mean", dist.stats["close"]["pct_10"])

        entry_price = current_pos.get("entry_price", current_price)
        pos_direction = current_pos.get("direction")

        if pos_direction == Direction.LONG and pred_high < entry_price:
            return Signal(
                direction=Direction.FLAT, size=0.0, entry=current_price,
                stop=0.0, target=0.0, strategy_name=self.name,
                confidence=1.0, expected_value=0.0,
                metadata={"action": "close_overnight", "reason": "range_below_entry"}
            )
        elif pos_direction == Direction.SHORT and pred_low > entry_price:
            return Signal(
                direction=Direction.FLAT, size=0.0, entry=current_price,
                stop=0.0, target=0.0, strategy_name=self.name,
                confidence=1.0, expected_value=0.0,
                metadata={"action": "close_overnight", "reason": "range_above_entry"}
            )

        return self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)


class RSIDivergenceStrategy(Strategy):
    """
    Detects RSI divergence confirmed by Kairos prediction direction.
    Bullish: price makes lower low but RSI makes higher low.
    Bearish: price makes higher high but RSI makes lower high.
    """
    name = "rsi_divergence"

    def __init__(self, rsi_period: int = 14, lookback_bars: int = 20,
                 divergence_threshold: float = 2.0):
        self.rsi_period = rsi_period
        self.lookback_bars = lookback_bars
        self.divergence_threshold = divergence_threshold

    def _rsi(self, close: np.ndarray) -> float:
        """Calculate RSI from close prices."""
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

    def _find_pivots(self, arr: np.ndarray, order: int = 2) -> List[int]:
        """Find local extrema indices."""
        if len(arr) < 2 * order + 1:
            return []
        pivots = []
        for i in range(order, len(arr) - order):
            if arr[i] > arr[i - order] and arr[i] > arr[i + order]:
                pivots.append(i)
            elif arr[i] < arr[i - order] and arr[i] < arr[i + order]:
                pivots.append(i)
        return pivots

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        if len(history) < max(self.lookback_bars, self.rsi_period + 5):
            return None

        close = history["close"].values[-self.lookback_bars:]
        rsi_values = np.array([
            self._rsi(history["close"].values[max(0, i - self.rsi_period):i])
            for i in range(len(history) - self.lookback_bars + 1, len(history) + 1)
        ])

        price_pivots = self._find_pivots(close, order=2)
        rsi_pivots = self._find_pivots(rsi_values, order=2)

        if len(price_pivots) < 2 or len(rsi_pivots) < 2:
            return None

        # Check for bullish divergence
        last_price_low_idx = None
        prev_price_low_idx = None
        for idx in sorted(price_pivots, reverse=True):
            if close[idx] <= close[max(0, idx - 1)] and close[idx] <= close[min(len(close) - 1, idx + 1)]:
                if last_price_low_idx is None:
                    last_price_low_idx = idx
                elif prev_price_low_idx is None:
                    prev_price_low_idx = idx
                    break

        if last_price_low_idx is not None and prev_price_low_idx is not None:
            if (close[last_price_low_idx] < close[prev_price_low_idx] and
                rsi_values[last_price_low_idx] > rsi_values[prev_price_low_idx] + self.divergence_threshold):
                if dist.stats["close"]["mean"] > current_price:
                    s = dist.stats["close"]
                    ev = dist.expected_value(current_price, s["pct_90"], s["pct_10"])
                    if ev > 0:
                        return Signal(
                            direction=Direction.LONG, size=0.5, entry=current_price,
                            stop=s["pct_10"], target=s["pct_90"],
                            strategy_name=self.name, confidence=0.7,
                            expected_value=ev,
                        )

        # Check for bearish divergence
        last_price_high_idx = None
        prev_price_high_idx = None
        for idx in sorted(price_pivots, reverse=True):
            if close[idx] >= close[max(0, idx - 1)] and close[idx] >= close[min(len(close) - 1, idx + 1)]:
                if last_price_high_idx is None:
                    last_price_high_idx = idx
                elif prev_price_high_idx is None:
                    prev_price_high_idx = idx
                    break

        if last_price_high_idx is not None and prev_price_high_idx is not None:
            if (close[last_price_high_idx] > close[prev_price_high_idx] and
                rsi_values[last_price_high_idx] < rsi_values[prev_price_high_idx] - self.divergence_threshold):
                if dist.stats["close"]["mean"] < current_price:
                    s = dist.stats["close"]
                    ev = dist.expected_value(current_price, s["pct_10"], s["pct_90"])
                    if ev > 0:
                        return Signal(
                            direction=Direction.SHORT, size=0.5, entry=current_price,
                            stop=s["pct_90"], target=s["pct_10"],
                            strategy_name=self.name, confidence=0.7,
                            expected_value=ev,
                        )

        return None


class LeverageCalibrationStrategy(Strategy):
    """
    Wraps a base strategy and scales size by predicted volatility-based leverage.
    Lower volatility → higher leverage. Higher volatility → lower leverage.
    """
    name = "leverage_calibration"

    def __init__(self, base_strategy: Strategy,
                 leverage_tiers: Optional[List[Tuple[float, float]]] = None,
                 max_leverage: float = 5.0):
        self.base_strategy = base_strategy
        self.max_leverage = max_leverage
        if leverage_tiers is None:
            self.leverage_tiers = [
                (0.02, 5.0),
                (0.04, 3.0),
                (0.06, 2.0),
                (float("inf"), 1.0),
            ]
        else:
            self.leverage_tiers = leverage_tiers

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None or base_signal.direction == Direction.FLAT:
            return base_signal

        s = dist.stats["close"]
        pred_range_pct = (s["pct_90"] - s["pct_10"]) / current_price

        leverage = 1.0
        for vol_threshold, lev in self.leverage_tiers:
            if pred_range_pct <= vol_threshold:
                leverage = lev
                break

        base_signal.size = min(base_signal.size * leverage, self.max_leverage)
        return base_signal


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


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward(
    strategy_factory: Callable[[], Strategy],
    data: pd.DataFrame,
    predictor: KairosPredictor,
    train_days: int = 250,
    test_days: int = 60,
    step: int = 60,
    anchored: bool = False,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Walk-forward validation: roll non-overlapping test windows with fresh
    strategy instances per fold. Computes per-fold and aggregate metrics,
    including Deflated Sharpe Ratio (DSR) as overfitting score.

    Parameters
    ----------
    strategy_factory : Callable[[], Strategy]
        Factory function that returns a fresh Strategy instance per call.
    data : pd.DataFrame
        OHLCV data with DatetimeIndex.
    predictor : KairosPredictor
        Predictor instance for generating distributions.
    train_days : int
        Number of days in training window (default 250).
    test_days : int
        Number of days in test window (default 60).
    step : int
        Window step size (default 60, non-overlapping test windows).
    anchored : bool
        If True: expanding window (train start fixed at 0).
        If False: sliding window (train window also rolls).
    initial_capital : float
        Starting capital per fold.
    fee_pct : float
        Transaction fee per trade.
    slippage_pct : float
        Slippage per trade entry.
    random_seed : Optional[int]
        Fixed seed for reproducibility (affects numpy/pd random state).

    Returns
    -------
    Dict with keys:
        - "folds": List[Dict] per-fold results {fold_id, train_metrics, test_metrics}
        - "aggregate_metrics": Dict of aggregate metrics across all folds
        - "overfitting_score": DSR (Deflated Sharpe Ratio)
        - "is_sharpe_mean": In-sample (train) Sharpe mean
        - "oos_sharpe_mean": Out-of-sample (test) Sharpe mean
        - "sharpe_degradation": IS Sharpe - OOS Sharpe
    """
    if len(data) < train_days + test_days:
        raise ValueError(
            f"Data length ({len(data)}) must be >= train_days + test_days ({train_days + test_days})"
        )
    if step <= 0:
        raise ValueError(f"step must be > 0 (got {step})")

    # Seed only the local scope: save/restore the global numpy RNG state so
    # that walk_forward()'s reproducibility guarantee doesn't leak into
    # unrelated code (e.g. other tests) that rely on unseeded np.random.
    _prev_random_state = np.random.get_state() if random_seed is not None else None
    if random_seed is not None:
        np.random.seed(random_seed)

    try:
        return _walk_forward_impl(
            strategy_factory, data, predictor, train_days, test_days, step,
            anchored, initial_capital, fee_pct, slippage_pct,
        )
    finally:
        if _prev_random_state is not None:
            np.random.set_state(_prev_random_state)


def _walk_forward_impl(
    strategy_factory, data, predictor, train_days, test_days, step,
    anchored, initial_capital, fee_pct, slippage_pct,
):
    folds = []
    fold_id = 0
    train_start = 0
    train_end = train_start + train_days

    while True:
        test_start = train_end
        test_end = test_start + test_days

        if test_end > len(data):
            break

        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]

        # Strategy factory: fresh instance per fold
        strategy = strategy_factory()

        # In-sample (training) backtest
        engine_is = BacktestEngine(
            predictor=predictor,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        router_is = DecisionTreeRouter(
            entropy_threshold=999.0,
            custom_strategies={
                "range": [strategy],
                "trend": [strategy],
                "uncertain": [strategy],
            },
        )
        train_metrics = engine_is.run(train_data, router_is, lookback=10)

        # Out-of-sample (test) backtest: fresh strategy instance
        strategy_oos = strategy_factory()
        engine_oos = BacktestEngine(
            predictor=predictor,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        router_oos = DecisionTreeRouter(
            entropy_threshold=999.0,
            custom_strategies={
                "range": [strategy_oos],
                "trend": [strategy_oos],
                "uncertain": [strategy_oos],
            },
        )
        test_metrics = engine_oos.run(test_data, router_oos, lookback=10)

        fold_result = {
            "fold_id": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        folds.append(fold_result)

        # Move to next fold. `train_end` (not `train_start`) is the loop
        # variable that must strictly increase every iteration so the loop
        # is guaranteed to terminate once test_end exceeds len(data).
        if anchored:
            # Expanding window: train_start stays 0, train_end grows by step.
            train_start = 0
            train_end = train_end + step
        else:
            # Sliding window: entire window rolls forward by step.
            train_start = train_start + step
            train_end = train_start + train_days

        fold_id += 1

    # Aggregate metrics
    is_sharpes = [f["train_metrics"]["sharpe"] for f in folds]
    oos_sharpes = [f["test_metrics"]["sharpe"] for f in folds]

    is_sharpe_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    oos_sharpe_mean = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    sharpe_degradation = is_sharpe_mean - oos_sharpe_mean

    # Deflated Sharpe Ratio (DSR)
    # Simple formula: DSR = OOS_Sharpe * (1 - exp(-degradation / max(OOS_Sharpe, 0.01)))
    # If OOS Sharpe is high and degradation is low, DSR approaches OOS Sharpe.
    # If degradation is high (overfitting), DSR is penalized.
    if oos_sharpe_mean > 0 and is_sharpe_mean > 0:
        dsr = oos_sharpe_mean * (
            1.0 - np.exp(-sharpe_degradation / max(oos_sharpe_mean, 0.01))
        )
    else:
        dsr = oos_sharpe_mean

    # Aggregate metrics: mean, std, median across folds
    all_train_metrics = [f["train_metrics"] for f in folds]
    all_test_metrics = [f["test_metrics"] for f in folds]

    def aggregate_fold_metrics(metrics_list):
        """Compute aggregate stats across folds."""
        keys = [
            "total_return",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "num_trades",
            "avg_trade",
            "avg_win",
            "avg_loss",
        ]
        result = {}
        for key in keys:
            values = [m.get(key, 0.0) for m in metrics_list]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
            result[f"{key}_median"] = float(np.median(values))
        return result

    aggregate_train = aggregate_fold_metrics(all_train_metrics)
    aggregate_test = aggregate_fold_metrics(all_test_metrics)

    return {
        "folds": folds,
        "num_folds": len(folds),
        "aggregate_train": aggregate_train,
        "aggregate_test": aggregate_test,
        "is_sharpe_mean": is_sharpe_mean,
        "oos_sharpe_mean": oos_sharpe_mean,
        "sharpe_degradation": sharpe_degradation,
        "overfitting_score": float(dsr),
    }
