"""
kairos_execution.py
===================
Path-dependent dynamic exits (#4) and Volume/Amount integration (#5).

Features:
- Partial exit execution plans (scale out at predicted high, close, trail)
- Trailing stop management
- Volume/amount filtering and confirmation
- Predicted VWAP analysis
- Volume fade detection
- Extended backtest engine with partial exit support

Usage:
    from kairos_execution import (
        PathExecutionStrategy, VolumeConfirmationStrategy,
        LiquidityFilterStrategy, VolumeFadeStrategy,
        PartialExitBacktestEngine
    )

    # Path execution: 3-leg scale-out
    strategy = PathExecutionStrategy(
        leg1_pct=0.33, leg1_target="high_pct_90",
        leg2_pct=0.33, leg2_target="close_pct_75",
        leg3_pct=0.34, leg3_trail=True
    )

    # Volume-aware: only trade if predicted volume > historical 30th percentile
    strategy = LiquidityFilterStrategy(min_volume_percentile=30)

    # Volume fade: fade moves on declining predicted volume
    strategy = VolumeFadeStrategy()
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

# Import base classes (with fallback for standalone use)
try:
    from kairos_backtest import (
        KairosDistribution, KairosPredictor, Direction,
        Signal, Strategy, Trade, BacktestEngine
    )
except ImportError:
    class KairosDistribution:
        def __init__(self, predictions):
            self.predictions = predictions
            self.df = pd.concat(predictions, ignore_index=True)
            self.stats = {}
            for col in ["open", "high", "low", "close"]:
                if col in self.df.columns:
                    arr = self.df[col].values.astype(float)
                    self.stats[col] = {
                        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                        "pct_10": float(np.percentile(arr, 10)),
                        "pct_25": float(np.percentile(arr, 25)),
                        "pct_50": float(np.percentile(arr, 50)),
                        "pct_75": float(np.percentile(arr, 75)),
                        "pct_90": float(np.percentile(arr, 90)),
                    }
        def expected_value(self, entry, target, stop, col="close"):
            values = self.df[col].values.astype(float)
            p_win = float(np.mean(values >= target))
            p_loss = float(np.mean(values <= stop))
            win_r = target - entry
            loss_r = entry - stop
            return float(p_win * win_r + p_loss * -loss_r)
        def kelly_fraction(self, entry, target, stop, col="close"):
            values = self.df[col].values.astype(float)
            p_win = float(np.mean(values >= target))
            p_loss = float(np.mean(values <= stop))
            if p_loss == 0: return 1.0
            b = (target - entry) / (entry - stop)
            if b <= 0: return 0.0
            return float(max(0.0, min((p_win * b - p_loss) / b, 1.0)))

    class Direction:
        LONG = 1
        SHORT = -1
        FLAT = 0

    @dataclass
    class Signal:
        direction: Any = None
        size: float = 0.0
        entry: float = 0.0
        stop: float = 0.0
        target: float = 0.0
        strategy_name: str = ""
        confidence: float = 0.0
        expected_value: float = 0.0
        metadata: Dict = field(default_factory=dict)

    class Strategy:
        name = "base"
        def generate_signal(self, dist, current_price, history, context):
            raise NotImplementedError

    @dataclass
    class Trade:
        entry_date: Any = None
        exit_date: Any = None
        direction: Any = None
        entry_price: float = 0.0
        exit_price: float = 0.0
        size: float = 0.0
        pnl: float = 0.0
        pnl_pct: float = 0.0
        strategy_name: str = ""
        exit_reason: str = ""

    class BacktestEngine:
        pass


# =============================================================================
# PART 1: PATH-DEPENDENT DYNAMIC EXITS
# =============================================================================

@dataclass
class ExitLeg:
    """One leg of a partial exit plan."""
    leg_id: int
    size_fraction: float  # 0.0 to 1.0, fraction of total position
    target_price: float
    stop_price: float
    trailing: bool = False
    trail_activation_price: Optional[float] = None  # when trailing activates
    trail_offset: Optional[float] = None  # distance behind peak to trail
    exit_on_hit: bool = True  # if False, leg stays open after target (rare)


@dataclass
class ExecutionPlan:
    """A complete entry + multi-leg exit plan."""
    direction: Direction
    total_size: float
    entry_price: float
    legs: List[ExitLeg]
    strategy_name: str
    confidence: float
    expected_value: float


class PathExecutionPlanner:
    """
    Builds execution plans from predicted paths.

    Default plan (3 legs):
    - Leg 1: 33% at predicted high (or 90th pct of high dist)
    - Leg 2: 33% at predicted close (or 75th pct of close dist)
    - Leg 3: 34% with trailing stop starting at predicted 50th pct,
             activating once price reaches 75th pct of the move
    """

    def __init__(self,
                 leg1_pct: float = 0.33,
                 leg1_target: str = "high_pct_90",
                 leg2_pct: float = 0.33,
                 leg2_target: str = "close_pct_75",
                 leg3_pct: float = 0.34,
                 leg3_trail: bool = True,
                 leg3_trail_activation: str = "close_pct_75",
                 leg3_trail_offset: str = "close_pct_50"):
        self.leg1_pct = leg1_pct
        self.leg1_target = leg1_target
        self.leg2_pct = leg2_pct
        self.leg2_target = leg2_target
        self.leg3_pct = leg3_pct
        self.leg3_trail = leg3_trail
        self.leg3_trail_activation = leg3_trail_activation
        self.leg3_trail_offset = leg3_trail_offset

    def _resolve_price(self, dist: KairosDistribution, spec: str) -> float:
        """Resolve a price spec like 'high_pct_90' to a float."""
        parts = spec.split("_")
        if len(parts) == 3 and parts[1] == "pct":
            col = parts[0]
            pct = int(parts[2])
            return dist.stats[col][f"pct_{pct}"]
        elif spec == "mean":
            return dist.stats["close"]["mean"]
        return 0.0

    def build_plan(self, dist: KairosDistribution, current_price: float,
                   direction: Direction) -> Optional[ExecutionPlan]:
        s_close = dist.stats["close"]
        s_high = dist.stats["high"]
        s_low = dist.stats["low"]

        if direction == Direction.LONG:
            entry = current_price
            hard_stop = s_low["pct_10"]

            t1 = self._resolve_price(dist, self.leg1_target)
            t2 = self._resolve_price(dist, self.leg2_target)

            legs = [
                ExitLeg(1, self.leg1_pct, t1, hard_stop),
                ExitLeg(2, self.leg2_pct, t2, hard_stop),
            ]

            if self.leg3_trail:
                activation = self._resolve_price(dist, self.leg3_trail_activation)
                offset_price = self._resolve_price(dist, self.leg3_trail_offset)
                trail_offset = activation - offset_price
                legs.append(ExitLeg(
                    3, self.leg3_pct, float("inf"), hard_stop,
                    trailing=True, trail_activation_price=activation,
                    trail_offset=trail_offset
                ))
            else:
                legs.append(ExitLeg(3, self.leg3_pct, t2, hard_stop))

        else:  # SHORT
            entry = current_price
            hard_stop = s_high["pct_90"]

            t1 = self._resolve_price(dist, self.leg1_target)
            t2 = self._resolve_price(dist, self.leg2_target)

            legs = [
                ExitLeg(1, self.leg1_pct, t1, hard_stop),
                ExitLeg(2, self.leg2_pct, t2, hard_stop),
            ]

            if self.leg3_trail:
                activation = self._resolve_price(dist, self.leg3_trail_activation)
                offset_price = self._resolve_price(dist, self.leg3_trail_offset)
                trail_offset = offset_price - activation  # reversed for short
                legs.append(ExitLeg(
                    3, self.leg3_pct, float("-inf"), hard_stop,
                    trailing=True, trail_activation_price=activation,
                    trail_offset=trail_offset
                ))
            else:
                legs.append(ExitLeg(3, self.leg3_pct, t2, hard_stop))

        # Compute expected value across all legs
        total_ev = 0.0
        for leg in legs:
            if not leg.trailing:
                ev = dist.expected_value(entry, leg.target_price, leg.stop_price)
                total_ev += ev * leg.size_fraction

        return ExecutionPlan(
            direction=direction,
            total_size=0.0,  # filled by strategy
            entry_price=entry,
            legs=legs,
            strategy_name="path_execution",
            confidence=0.0,  # filled by strategy
            expected_value=total_ev
        )


class PathExecutionStrategy(Strategy):
    """
    Strategy that uses path-dependent partial exits.
    Entry is directional based on predicted close mean.
    """
    name = "path_execution"

    def __init__(self,
                 leg1_pct: float = 0.33,
                 leg1_target: str = "high_pct_90",
                 leg2_pct: float = 0.33,
                 leg2_target: str = "close_pct_75",
                 leg3_pct: float = 0.34,
                 leg3_trail: bool = True):
        self.planner = PathExecutionPlanner(
            leg1_pct, leg1_target, leg2_pct, leg2_target,
            leg3_pct, leg3_trail
        )

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        mean = s["mean"]

        if mean > current_price:
            direction = Direction.LONG
        elif mean < current_price:
            direction = Direction.SHORT
        else:
            return None

        plan = self.planner.build_plan(dist, current_price, direction)
        if plan is None or plan.expected_value <= 0:
            return None

        stop = dist.stats["low"]["pct_10"] if direction == Direction.LONG else dist.stats["high"]["pct_90"]
        target = dist.stats["high"]["pct_90"] if direction == Direction.LONG else dist.stats["low"]["pct_10"]

        kelly = dist.kelly_fraction(current_price, target, stop)
        plan.total_size = min(kelly * 0.5, 1.0)

        # Confidence based on predicted Sharpe
        sharpe = abs(mean - current_price) / s["std"] if s["std"] > 0 else 0.0
        plan.confidence = min(sharpe, 1.0)

        return Signal(
            direction=direction,
            size=plan.total_size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=plan.confidence,
            expected_value=plan.expected_value,
            metadata={"execution_plan": plan}
        )


class PathHighLowExecutionStrategy(Strategy):
    """
    Uses the strongest signal (predicted high/low) for partial exits.
    Leg 1: 50% at predicted high
    Leg 2: 50% at predicted close
    """
    name = "path_high_low_execution"

    def __init__(self, leg1_pct: float = 0.5, leg2_pct: float = 0.5):
        self.leg1_pct = leg1_pct
        self.leg2_pct = leg2_pct

    def generate_signal(self, dist, current_price, history, context):
        h_mean = dist.stats["high"]["mean"]
        l_mean = dist.stats["low"]["mean"]
        c_mean = dist.stats["close"]["mean"]

        if c_mean > current_price:
            direction = Direction.LONG
            entry = current_price
            leg1_target = h_mean
            leg2_target = c_mean
            stop = l_mean * 0.98
        elif c_mean < current_price:
            direction = Direction.SHORT
            entry = current_price
            leg1_target = l_mean
            leg2_target = c_mean
            stop = h_mean * 1.02
        else:
            return None

        plan = ExecutionPlan(
            direction=direction,
            total_size=0.0,
            entry_price=entry,
            legs=[
                ExitLeg(1, self.leg1_pct, leg1_target, stop),
                ExitLeg(2, self.leg2_pct, leg2_target, stop),
            ],
            strategy_name=self.name,
            confidence=0.0,
            expected_value=0.0
        )

        ev1 = dist.expected_value(entry, leg1_target, stop)
        ev2 = dist.expected_value(entry, leg2_target, stop)
        plan.expected_value = ev1 * self.leg1_pct + ev2 * self.leg2_pct

        if plan.expected_value <= 0:
            return None

        kelly = dist.kelly_fraction(entry, leg1_target, stop)
        plan.total_size = min(kelly * 0.5, 1.0)
        plan.confidence = abs(c_mean - current_price) / current_price / (dist.stats["close"]["std"] / current_price + 0.001)

        return Signal(
            direction=direction,
            size=plan.total_size,
            entry=current_price,
            stop=stop,
            target=leg1_target,
            strategy_name=self.name,
            confidence=min(plan.confidence, 1.0),
            expected_value=plan.expected_value,
            metadata={"execution_plan": plan}
        )


# =============================================================================
# PART 2: VOLUME / AMOUNT INTEGRATION
# =============================================================================

class VolumeAnalyzer:
    """
    Analyzes predicted volume and amount from the Kairos distribution.
    """

    def __init__(self, dist: KairosDistribution):
        self.dist = dist
        self.vol_stats = self._compute_vol_stats()
        self.amount_stats = self._compute_amount_stats()

    def _compute_vol_stats(self) -> Dict[str, float]:
        if "volume" not in self.dist.df.columns:
            return {}
        arr = self.dist.df["volume"].values.astype(float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "pct_10": float(np.percentile(arr, 10)),
            "pct_25": float(np.percentile(arr, 25)),
            "pct_50": float(np.percentile(arr, 50)),
            "pct_75": float(np.percentile(arr, 75)),
            "pct_90": float(np.percentile(arr, 90)),
        }

    def _compute_amount_stats(self) -> Dict[str, float]:
        if "amount" not in self.dist.df.columns:
            return {}
        arr = self.dist.df["amount"].values.astype(float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "pct_10": float(np.percentile(arr, 10)),
            "pct_50": float(np.percentile(arr, 50)),
            "pct_90": float(np.percentile(arr, 90)),
        }

    def predicted_volume(self) -> float:
        return self.vol_stats.get("mean", 0.0)

    def predicted_amount(self) -> float:
        return self.amount_stats.get("mean", 0.0)

    def volume_vs_history(self, history: pd.DataFrame) -> Dict[str, float]:
        """Compare predicted volume to historical distribution."""
        if "volume" not in history.columns or not self.vol_stats:
            return {"percentile": 50.0, "ratio": 1.0}

        hist_vol = history["volume"].values.astype(float)
        pred_vol = self.predicted_volume()

        percentile = float(stats.percentileofscore(hist_vol, pred_vol)) if len(hist_vol) > 0 else 50.0
        median_hist = float(np.median(hist_vol)) if len(hist_vol) > 0 else pred_vol
        ratio = pred_vol / median_hist if median_hist > 0 else 1.0

        return {"percentile": percentile, "ratio": ratio}

    def amount_vs_history(self, history: pd.DataFrame) -> Dict[str, float]:
        """Compare predicted amount to historical distribution."""
        if "amount" not in history.columns or not self.amount_stats:
            return {"percentile": 50.0, "ratio": 1.0}

        hist_amount = history["amount"].values.astype(float)
        pred_amount = self.predicted_amount()

        percentile = float(stats.percentileofscore(hist_amount, pred_amount)) if len(hist_amount) > 0 else 50.0
        median_hist = float(np.median(hist_amount)) if len(hist_amount) > 0 else pred_amount
        ratio = pred_amount / median_hist if median_hist > 0 else 1.0

        return {"percentile": percentile, "ratio": ratio}

    def is_liquid(self, history: pd.DataFrame, min_percentile: float = 30.0) -> bool:
        v = self.volume_vs_history(history)
        return v["percentile"] >= min_percentile

    def volume_confirms_direction(self, history: pd.DataFrame) -> Optional[bool]:
        """
        Returns True if predicted volume is above median AND direction is clear.
        Returns False if predicted volume is below median (fade signal).
        Returns None if ambiguous.
        """
        v = self.volume_vs_history(history)
        if v["percentile"] >= 60.0:
            return True
        elif v["percentile"] <= 30.0:
            return False
        return None

    def amount_flow_direction(self) -> Optional[int]:
        """
        Returns 1 if predicted amount is strongly positive (inflow),
        -1 if strongly negative (outflow), 0 if neutral.
        """
        if not self.amount_stats:
            return None
        mean = self.amount_stats["mean"]
        std = self.amount_stats["std"]
        if std == 0:
            return 0
        z = mean / std
        if z > 1.0:
            return 1
        elif z < -1.0:
            return -1
        return 0

    def predicted_vwap(self) -> Optional[float]:
        """Approximate predicted VWAP = predicted amount / predicted volume."""
        pv = self.predicted_volume()
        pa = self.predicted_amount()
        if pv > 0:
            return pa / pv
        return None


class LiquidityFilterStrategy(Strategy):
    """
    Only passes signals if predicted volume is above a historical percentile.
    Acts as a wrapper around a base strategy.
    """
    name = "liquidity_filter"

    def __init__(self, base_strategy: Strategy, min_volume_percentile: float = 30.0):
        self.base_strategy = base_strategy
        self.min_volume_percentile = min_volume_percentile

    def generate_signal(self, dist, current_price, history, context):
        analyzer = VolumeAnalyzer(dist)
        if not analyzer.is_liquid(history, self.min_volume_percentile):
            return None

        signal = self.base_strategy.generate_signal(dist, current_price, history, context)
        if signal is None:
            return None

        # Augment metadata; preserve the inner strategy's name for tracking
        v = analyzer.volume_vs_history(history)
        signal.metadata["volume_percentile"] = v["percentile"]
        signal.metadata["volume_ratio"] = v["ratio"]
        return signal


class VolumeConfirmationStrategy(Strategy):
    """
    Only takes directional trades if predicted volume confirms the move.
    Up move + high volume = go long.
    Up move + low volume = fade (short).
    """
    name = "volume_confirmation"

    def __init__(self, volume_threshold_pct: float = 60.0, fade_threshold_pct: float = 30.0):
        self.volume_threshold_pct = volume_threshold_pct
        self.fade_threshold_pct = fade_threshold_pct

    def generate_signal(self, dist, current_price, history, context):
        analyzer = VolumeAnalyzer(dist)
        v = analyzer.volume_vs_history(history)

        s = dist.stats["close"]
        mean = s["mean"]

        if mean > current_price:
            base_direction = Direction.LONG
        elif mean < current_price:
            base_direction = Direction.SHORT
        else:
            return None

        # Volume confirmation logic
        if v["percentile"] >= self.volume_threshold_pct:
            # High volume confirms the move
            direction = base_direction
        elif v["percentile"] <= self.fade_threshold_pct:
            # Low volume: fade the move
            direction = Direction.SHORT if base_direction == Direction.LONG else Direction.LONG
        else:
            return None  # Ambiguous volume

        if direction == Direction.LONG:
            stop = dist.stats["low"]["pct_10"]
            target = dist.stats["high"]["pct_90"]
        else:
            stop = dist.stats["high"]["pct_90"]
            target = dist.stats["low"]["pct_10"]

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
            confidence=min(v["percentile"] / 100.0, 1.0),
            expected_value=ev,
            metadata={
                "volume_percentile": v["percentile"],
                "volume_ratio": v["ratio"],
                "base_direction": base_direction.value,
            }
        )


class VolumeFadeStrategy(Strategy):
    """
    Specifically fades moves that occur on declining predicted volume.
    A pure contrarian volume strategy.
    """
    name = "volume_fade"

    def __init__(self, max_volume_percentile: float = 35.0, min_move_pct: float = 0.005):
        self.max_volume_percentile = max_volume_percentile
        self.min_move_pct = min_move_pct

    def generate_signal(self, dist, current_price, history, context):
        analyzer = VolumeAnalyzer(dist)
        v = analyzer.volume_vs_history(history)

        if v["percentile"] > self.max_volume_percentile:
            return None

        s = dist.stats["close"]
        move = (s["mean"] - current_price) / current_price
        if abs(move) < self.min_move_pct:
            return None

        # Fade the move: if predicted up on low volume, short
        if move > 0:
            direction = Direction.SHORT
            stop = dist.stats["high"]["pct_95"]
            target = dist.stats["low"]["pct_50"]
        else:
            direction = Direction.LONG
            stop = dist.stats["low"]["pct_5"]
            target = dist.stats["high"]["pct_50"]

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
            confidence=1.0 - v["percentile"] / self.max_volume_percentile,
            expected_value=ev,
            metadata={
                "volume_percentile": v["percentile"],
                "predicted_move": move,
            }
        )


class AmountFlowStrategy(Strategy):
    """
    Uses predicted amount (notional flow) as a directional signal.
    Strong inflow + up close = accumulation (long).
    Strong outflow + up close = distribution (short).
    """
    name = "amount_flow"

    def __init__(self, z_threshold: float = 1.0):
        self.z_threshold = z_threshold

    def generate_signal(self, dist, current_price, history, context):
        analyzer = VolumeAnalyzer(dist)
        flow = analyzer.amount_flow_direction()

        if flow is None:
            return None

        s = dist.stats["close"]
        mean = s["mean"]
        close_dir = 1 if mean > current_price else (-1 if mean < current_price else 0)

        if close_dir == 0:
            return None

        # Inflow + up = long, Outflow + up = short (distribution)
        if flow == 1 and close_dir == 1:
            direction = Direction.LONG
        elif flow == -1 and close_dir == 1:
            direction = Direction.SHORT
        elif flow == 1 and close_dir == -1:
            direction = Direction.LONG  # accumulation on dip
        elif flow == -1 and close_dir == -1:
            direction = Direction.SHORT  # distribution on decline
        else:
            return None

        if direction == Direction.LONG:
            stop = dist.stats["low"]["pct_10"]
            target = dist.stats["high"]["pct_90"]
        else:
            stop = dist.stats["high"]["pct_90"]
            target = dist.stats["low"]["pct_10"]

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
            confidence=abs(flow) * abs(close_dir),
            expected_value=ev,
            metadata={
                "amount_flow": flow,
                "close_direction": close_dir,
            }
        )


class PredictedVWAPStrategy(Strategy):
    """
    Compare predicted close to predicted VWAP.
    Close > VWAP = bullish (accumulation day)
    Close < VWAP = bearish (distribution day)
    """
    name = "predicted_vwap"

    def __init__(self, min_deviation_pct: float = 0.001):
        self.min_deviation_pct = min_deviation_pct

    def generate_signal(self, dist, current_price, history, context):
        analyzer = VolumeAnalyzer(dist)
        vwap = analyzer.predicted_vwap()
        if vwap is None:
            return None

        s = dist.stats["close"]
        pred_close = s["mean"]
        deviation = (pred_close - vwap) / vwap

        if abs(deviation) < self.min_deviation_pct:
            return None

        if deviation > 0:
            direction = Direction.LONG
            stop = dist.stats["low"]["pct_10"]
            target = dist.stats["high"]["pct_90"]
        else:
            direction = Direction.SHORT
            stop = dist.stats["high"]["pct_90"]
            target = dist.stats["low"]["pct_10"]

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
            confidence=min(abs(deviation) / self.min_deviation_pct, 1.0),
            expected_value=ev,
            metadata={
                "predicted_vwap": vwap,
                "vwap_deviation": deviation,
            }
        )


# =============================================================================
# PART 3: PARTIAL EXIT BACKTEST ENGINE
# =============================================================================

@dataclass
class ActiveLeg:
    """Tracks an open leg during backtesting."""
    leg_id: int
    direction: Direction
    size: float
    entry_price: float
    target_price: float
    stop_price: float
    trailing: bool
    trail_activation_price: Optional[float]
    trail_offset: Optional[float]
    highest_price: float  # for trailing stop tracking
    lowest_price: float
    is_open: bool = True
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


class PartialExitBacktestEngine(BacktestEngine):
    """
    Extended backtest engine that supports partial exits via ExecutionPlan.
    """

    def __init__(self,
                 predictor: Any,
                 fee_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 initial_capital: float = 10000.0):
        self.predictor = predictor
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.initial_capital = initial_capital

        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.signals: List[Tuple[Any, Signal]] = []
        self.active_legs: List[ActiveLeg] = []

    def run(self, df: pd.DataFrame, router: Any, lookback: int = 100) -> Dict:
        capital = self.initial_capital
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

            # Manage existing legs
            open_legs = []
            for leg in self.active_legs:
                if not leg.is_open:
                    continue

                exit_price, exit_reason = self._check_leg_exit(leg, tomorrow)
                if exit_price is not None:
                    pnl = self._calculate_leg_pnl(leg, exit_price)
                    capital += pnl
                    leg.exit_price = exit_price
                    leg.exit_reason = exit_reason
                    leg.is_open = False
                    self.trades.append(Trade(
                        entry_date=date,  # simplified
                        exit_date=df.index[i + 1],
                        direction=leg.direction,
                        entry_price=leg.entry_price,
                        exit_price=exit_price,
                        size=leg.size,
                        pnl=pnl,
                        pnl_pct=(pnl / (leg.entry_price * leg.size)) if leg.entry_price * leg.size != 0 else 0.0,
                        strategy_name="partial_exit",
                        exit_reason=exit_reason,
                    ))
                else:
                    # Update trailing stop tracking
                    high = float(tomorrow["high"])
                    low = float(tomorrow["low"])
                    if high > leg.highest_price:
                        leg.highest_price = high
                    if low < leg.lowest_price:
                        leg.lowest_price = low

                    # Update trailing stop if activated
                    if leg.trailing and leg.trail_activation_price is not None:
                        if leg.direction == Direction.LONG:
                            if leg.highest_price >= leg.trail_activation_price:
                                new_stop = leg.highest_price - (leg.trail_offset or 0)
                                if new_stop > leg.stop_price:
                                    leg.stop_price = new_stop
                        else:
                            if leg.lowest_price <= leg.trail_activation_price:
                                new_stop = leg.lowest_price + (leg.trail_offset or 0)
                                if new_stop < leg.stop_price:
                                    leg.stop_price = new_stop

                    open_legs.append(leg)

            self.active_legs = open_legs

            # Enter new legs from signal
            if signal and signal.direction != Direction.FLAT:
                plan = signal.metadata.get("execution_plan") if hasattr(signal, "metadata") else None

                if plan and isinstance(plan, ExecutionPlan):
                    # Multi-leg entry
                    entry_price = float(tomorrow["open"]) * (
                        1.0 + self.slippage_pct * signal.direction.value
                    )
                    notional = signal.size * capital
                    fee = notional * self.fee_pct
                    capital -= fee

                    for leg_def in plan.legs:
                        leg_size = notional * leg_def.size_fraction / entry_price
                        leg = ActiveLeg(
                            leg_id=leg_def.leg_id,
                            direction=signal.direction,
                            size=leg_size,
                            entry_price=entry_price,
                            target_price=leg_def.target_price,
                            stop_price=leg_def.stop_price,
                            trailing=leg_def.trailing,
                            trail_activation_price=leg_def.trail_activation_price,
                            trail_offset=leg_def.trail_offset,
                            highest_price=entry_price,
                            lowest_price=entry_price,
                        )
                        self.active_legs.append(leg)
                else:
                    # Single-leg entry (fallback to standard behavior)
                    entry_price = float(tomorrow["open"]) * (
                        1.0 + self.slippage_pct * signal.direction.value
                    )
                    notional = signal.size * capital
                    fee = notional * self.fee_pct
                    capital -= fee

                    leg = ActiveLeg(
                        leg_id=0,
                        direction=signal.direction,
                        size=notional / entry_price,
                        entry_price=entry_price,
                        target_price=signal.target,
                        stop_price=signal.stop,
                        trailing=False,
                        trail_activation_price=None,
                        trail_offset=None,
                        highest_price=entry_price,
                        lowest_price=entry_price,
                    )
                    self.active_legs.append(leg)

            self.equity_curve.append((date, capital))
            prev_capital = capital

        # Close remaining legs at last close
        if self.active_legs and len(df) > 0:
            last_close = float(df.iloc[-1]["close"])
            for leg in self.active_legs:
                if leg.is_open:
                    pnl = self._calculate_leg_pnl(leg, last_close)
                    capital += pnl
                    leg.exit_price = last_close
                    leg.exit_reason = "end_of_data"
                    leg.is_open = False
                    self.trades.append(Trade(
                        entry_date=df.index[-1],
                        exit_date=df.index[-1],
                        direction=leg.direction,
                        entry_price=leg.entry_price,
                        exit_price=last_close,
                        size=leg.size,
                        pnl=pnl,
                        pnl_pct=(pnl / (leg.entry_price * leg.size)) if leg.entry_price * leg.size != 0 else 0.0,
                        strategy_name="partial_exit",
                        exit_reason="end_of_data",
                    ))
            self.equity_curve[-1] = (self.equity_curve[-1][0], capital)
            self.active_legs = []

        return self._compute_metrics()

    def _check_leg_exit(self, leg: ActiveLeg, tomorrow: pd.Series) -> Tuple[Optional[float], Optional[str]]:
        open_p = float(tomorrow["open"])
        high = float(tomorrow["high"])
        low = float(tomorrow["low"])
        close = float(tomorrow["close"])

        if leg.direction == Direction.LONG:
            # Check stop first
            if open_p <= leg.stop_price:
                return open_p, f"leg{leg.leg_id}_stop_open"
            if low <= leg.stop_price:
                return leg.stop_price, f"leg{leg.leg_id}_stop"

            # Check target (unless trailing leg with no fixed target)
            if not leg.trailing and open_p >= leg.target_price:
                return open_p, f"leg{leg.leg_id}_target_open"
            if not leg.trailing and high >= leg.target_price:
                return leg.target_price, f"leg{leg.leg_id}_target"

            # Trailing leg: check if trailing stop hit
            if leg.trailing:
                if open_p <= leg.stop_price:
                    return open_p, f"leg{leg.leg_id}_trail_stop"
                if low <= leg.stop_price:
                    return leg.stop_price, f"leg{leg.leg_id}_trail_stop"

            return None, None

        else:  # SHORT
            if open_p >= leg.stop_price:
                return open_p, f"leg{leg.leg_id}_stop_open"
            if high >= leg.stop_price:
                return leg.stop_price, f"leg{leg.leg_id}_stop"

            if not leg.trailing and open_p <= leg.target_price:
                return open_p, f"leg{leg.leg_id}_target_open"
            if not leg.trailing and low <= leg.target_price:
                return leg.target_price, f"leg{leg.leg_id}_target"

            if leg.trailing:
                if open_p >= leg.stop_price:
                    return open_p, f"leg{leg.leg_id}_trail_stop"
                if high >= leg.stop_price:
                    return leg.stop_price, f"leg{leg.leg_id}_trail_stop"

            return None, None

    def _calculate_leg_pnl(self, leg: ActiveLeg, exit_price: float) -> float:
        if leg.direction == Direction.LONG:
            gross = (exit_price - leg.entry_price) * leg.size
        else:
            gross = (leg.entry_price - exit_price) * leg.size
        fee = exit_price * leg.size * self.fee_pct
        return gross - fee

    def _compute_metrics(self) -> Dict:
        if not self.trades:
            return {
                "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0,
                "avg_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
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


# =============================================================================
# PART 4: EXECUTION WRAPPERS
# =============================================================================

class PyramidingStrategy(Strategy):
    """
    Wraps a base strategy and encodes a pyramid add plan in signal metadata.

    The strategy itself does NOT manage positions - it generates an initial signal
    and encodes a pyramid plan in metadata. The orchestrator is responsible for
    executing adds based on this plan.

    Pyramid plan is a list of dicts, each containing:
    - level: int (1, 2, 3, ...)
    - add_at_price: float (price level at which to add)
    - add_size: float (size to add at this level)
    - stop: float (stop loss for the full position)
    - target: float (target price for the full position)
    """
    name = "pyramiding"

    def __init__(self, base_strategy: Strategy,
                 pyramid_threshold_pct: float = 0.01,
                 pyramid_add_pct: float = 0.25,
                 max_pyramid_levels: int = 3):
        self.base_strategy = base_strategy
        self.pyramid_threshold_pct = pyramid_threshold_pct
        self.pyramid_add_pct = pyramid_add_pct
        self.max_pyramid_levels = max_pyramid_levels

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        # If direction is FLAT, return as-is without pyramid plan
        if base_signal.direction == Direction.FLAT:
            return base_signal

        # Build pyramid plan
        direction = base_signal.direction
        base_size = base_signal.size
        entry_price = base_signal.entry

        pyramid_plan = []
        for level in range(1, self.max_pyramid_levels + 1):
            if direction == Direction.LONG:
                add_at_price = entry_price * (1.0 + self.pyramid_threshold_pct * level)
            else:  # SHORT
                add_at_price = entry_price * (1.0 - self.pyramid_threshold_pct * level)

            pyramid_plan.append({
                "level": level,
                "add_at_price": add_at_price,
                "add_size": base_size * self.pyramid_add_pct,
                "stop": base_signal.stop,
                "target": base_signal.target,
            })

        base_signal.metadata["pyramid_plan"] = pyramid_plan
        return base_signal


class TimeBasedStopStrategy(Strategy):
    """
    Wraps a base strategy and annotates signal with time-exit metadata.

    Adds time-exit information to the signal. The orchestrator checks this
    metadata when managing positions to exit if the target isn't reached
    by time_exit_bar.

    Metadata keys added:
    - time_exit_enabled: bool (True if time exit is enabled)
    - time_exit_bar: int (bar index at which to exit if target not reached)
    - time_exit_price: float (price at which to exit on time)
    """
    name = "time_based_stop"

    def __init__(self, base_strategy: Strategy,
                 time_bars: int = 1,
                 exit_at: str = "close"):
        """
        Args:
            base_strategy: Strategy to wrap
            time_bars: Number of bars to wait before time exit
            exit_at: Where to exit on time - "close" or "predicted_median"
        """
        self.base_strategy = base_strategy
        self.time_bars = time_bars
        self.exit_at = exit_at

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        # Get current bar index from context (default to 0 if not provided)
        current_bar = context.get("bar_index", 0)

        # Determine time exit price
        if self.exit_at == "predicted_median":
            time_exit_price = dist.stats["close"]["pct_50"]
        else:  # "close"
            time_exit_price = current_price

        # Add time-exit metadata
        base_signal.metadata["time_exit_enabled"] = True
        base_signal.metadata["time_exit_bar"] = current_bar + self.time_bars
        base_signal.metadata["time_exit_price"] = time_exit_price

        return base_signal


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    print("kairos_execution.py loaded successfully.")
    print("Classes available:")
    print("  PathExecutionPlanner, PathExecutionStrategy, PathHighLowExecutionStrategy")
    print("  VolumeAnalyzer, LiquidityFilterStrategy, VolumeConfirmationStrategy")
    print("  VolumeFadeStrategy, AmountFlowStrategy, PredictedVWAPStrategy")
    print("  PartialExitBacktestEngine")
    print("  PyramidingStrategy, TimeBasedStopStrategy")
