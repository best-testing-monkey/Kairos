"""
kairos_horizon.py
=================
Multi-Horizon Prediction Stack for Kairos.

Predicts T+1, T+2, T+3 (and beyond) by iteratively chaining
single-horizon predictions. Analyzes confidence decay across horizons
to determine optimal hold periods.

Usage:
    from kairos_horizon import KairosMultiHorizonPredictor, HorizonStack

    predictor = KairosMultiHorizonPredictor(predict_kairos_cloud, max_horizon=3)
    stack = predictor.predict(history_df)

    # stack.horizons[1] is KairosDistribution for T+1
    # stack.horizons[2] is KairosDistribution for T+2
    # stack.horizons[3] is KairosDistribution for T+3

    if stack.hold_recommendation() == "hold_3_days":
        # T+1, T+2, T+3 all confirm direction with acceptable decay
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Import existing classes (assumes kairos_backtest.py is available)
try:
    from kairos_backtest import KairosDistribution, KairosPredictor, Direction, Signal, Strategy
except ImportError:
    # Fallback definitions for standalone use
    class KairosDistribution:
        def __init__(self, predictions):
            self.predictions = predictions
            self.df = pd.concat(predictions, ignore_index=True)
            self.stats = {}
            for col in ["open", "high", "low", "close"]:
                if col in self.df.columns:
                    arr = self.df[col].values.astype(float)
                    self.stats[col] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "pct_10": float(np.percentile(arr, 10)),
                        "pct_90": float(np.percentile(arr, 90)),
                        "pct_25": float(np.percentile(arr, 25)),
                        "pct_75": float(np.percentile(arr, 75)),
                    }

    class Direction:
        LONG = 1
        SHORT = -1
        FLAT = 0

    class Signal:
        pass

    class Strategy:
        name = "base"


# =============================================================================
# HORIZON STACK
# =============================================================================

@dataclass
class HorizonStack:
    """
    Holds distributions for T+1 through T+N, plus decay analysis.
    """
    horizons: Dict[int, KairosDistribution] = field(default_factory=dict)
    base_price: float = 0.0
    base_date: Optional[pd.Timestamp] = None

    # Decay metrics
    std_decay: Dict[int, float] = field(default_factory=dict)
    mean_drift: Dict[int, float] = field(default_factory=dict)
    directional_consistency: float = 0.0  # % of horizons agreeing with T+1
    sharpe_decay: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self._compute_decay()

    def _compute_decay(self):
        if not self.horizons or 1 not in self.horizons:
            return

        t1_mean = self.horizons[1].stats["close"]["mean"]
        t1_std = self.horizons[1].stats["close"]["std"]
        t1_sharpe = t1_mean / t1_std if t1_std > 0 else 0.0

        directions = []
        for h, dist in sorted(self.horizons.items()):
            s = dist.stats["close"]
            self.std_decay[h] = s["std"] / t1_std if t1_std > 0 else 1.0
            self.mean_drift[h] = (s["mean"] - t1_mean) / t1_std if t1_std > 0 else 0.0
            sharpe = s["mean"] / s["std"] if s["std"] > 0 else 0.0
            self.sharpe_decay[h] = sharpe / t1_sharpe if t1_sharpe != 0 else 0.0

            # Direction relative to base price
            if s["mean"] > self.base_price:
                directions.append(1)
            elif s["mean"] < self.base_price:
                directions.append(-1)
            else:
                directions.append(0)

        if directions:
            t1_dir = directions[0]
            self.directional_consistency = sum(1 for d in directions if d == t1_dir) / len(directions)

    def hold_recommendation(self,
                           max_std_multiplier: float = 2.5,
                           min_consistency: float = 0.67,
                           min_sharpe_ratio: float = 0.3) -> str:
        """
        Returns: "flat", "hold_1_day", "hold_2_days", or "hold_3_days"
        """
        if not self.horizons or 1 not in self.horizons:
            return "flat"

        t1 = self.horizons[1].stats["close"]
        if t1["std"] == 0:
            return "flat"

        # Check T+1 is viable
        t1_sharpe = abs(t1["mean"] - self.base_price) / t1["std"]
        if t1_sharpe < min_sharpe_ratio:
            return "flat"

        # Check subsequent horizons
        max_hold = 1
        for h in sorted(self.horizons.keys()):
            if h == 1:
                continue
            if h not in self.std_decay:
                break

            # Std must not blow up too much
            if self.std_decay[h] > max_std_multiplier:
                break

            # Direction must agree with T+1
            h_mean = self.horizons[h].stats["close"]["mean"]
            t1_dir = 1 if t1["mean"] > self.base_price else -1
            h_dir = 1 if h_mean > self.base_price else (-1 if h_mean < self.base_price else 0)
            if h_dir != t1_dir:
                break

            max_hold = h

        # Overall consistency check
        if self.directional_consistency < min_consistency:
            return "hold_1_day"

        return f"hold_{max_hold}_day{'s' if max_hold > 1 else ''}"

    def confidence_curve(self) -> List[Tuple[int, float]]:
        """Returns (horizon, confidence_score) pairs."""
        scores = []
        for h in sorted(self.horizons.keys()):
            s = self.horizons[h].stats["close"]
            if s["std"] > 0:
                sharpe = abs(s["mean"] - self.base_price) / s["std"]
                scores.append((h, sharpe))
        return scores

    def __repr__(self) -> str:
        lines = [
            f"HorizonStack(base_price={self.base_price:.4f}, horizons={list(self.horizons.keys())})",
            f"  Directional consistency: {self.directional_consistency:.2f}",
            f"  Std decay: {self.std_decay}",
            f"  Sharpe decay: {self.sharpe_decay}",
            f"  Recommendation: {self.hold_recommendation()}",
        ]
        return "\n".join(lines)


# =============================================================================
# MULTI-HORIZON PREDICTOR
# =============================================================================

class KairosMultiHorizonPredictor:
    """
    Chains single-horizon predictions to produce T+1, T+2, T+3 distributions.

    For each horizon h > 1:
    1. Take the median prediction from horizon h-1 as a synthetic bar
    2. Append it to history
    3. Call predict_kairos_cloud again
    4. Store the resulting distribution
    """

    def __init__(self,
                 predict_fn: Callable[[pd.DataFrame, ...], List[pd.DataFrame]],
                 max_horizon: int = 3,
                 synthetic_volume: Optional[str] = "median"):
        self.predict_fn = predict_fn
        self.max_horizon = max_horizon
        self.synthetic_volume = synthetic_volume

    def predict(self, history: pd.DataFrame, **kwargs) -> HorizonStack:
        base_price = float(history["close"].iloc[-1])
        base_date = history.index[-1]
        stack = HorizonStack(base_price=base_price, base_date=base_date)

        current_history = history.copy()

        for h in range(1, self.max_horizon + 1):
            # Predict next day
            predictions = self.predict_fn(current_history, **kwargs)
            try:
                from kairos_backtest import distribution_for
                dist = distribution_for(predictions)
            except ImportError:
                dist = KairosDistribution(predictions)
            stack.horizons[h] = dist

            # Build synthetic bar for next iteration
            if h < self.max_horizon:
                synthetic = self._build_synthetic_bar(dist, current_history)
                current_history = pd.concat([current_history, synthetic])

        return stack

    def _build_synthetic_bar(self, dist: KairosDistribution,
                             history: pd.DataFrame) -> pd.DataFrame:
        """
        Build a single synthetic DataFrame row from the median of the distribution.
        Uses the next calendar day as the index.
        """
        s = dist.stats

        # Use median for OHLC
        o = s["open"]["mean"] if "mean" in s.get("open", {}) else s["open"]["median"]
        h = s["high"]["mean"] if "mean" in s.get("high", {}) else s["high"]["median"]
        l = s["low"]["mean"] if "mean" in s.get("low", {}) else s["low"]["median"]
        c = s["close"]["mean"] if "mean" in s.get("close", {}) else s["close"]["median"]

        # Volume and amount: use historical median or predicted median
        if "volume" in s and "median" in s["volume"]:
            v = s["volume"]["median"]
        else:
            v = history["volume"].median() if "volume" in history.columns else 0.0

        if "amount" in s and "median" in s["amount"]:
            a = s["amount"]["median"]
        else:
            a = history["amount"].median() if "amount" in history.columns else 0.0

        # Next date
        last_date = history.index[-1]
        if isinstance(last_date, pd.Timestamp):
            next_date = last_date + pd.Timedelta(days=1)
        else:
            next_date = last_date + 1

        df = pd.DataFrame({
            "open": [o], "high": [h], "low": [l], "close": [c],
            "volume": [v], "amount": [a]
        }, index=[next_date])

        return df


# =============================================================================
# MULTI-HORIZON STRATEGIES
# =============================================================================

class MultiHorizonHoldStrategy(Strategy):
    """
    Uses the full horizon stack to decide how long to hold.

    Entry: T+1 open, direction from T+1 prediction.
    Hold: If T+2 and T+3 confirm direction and std decay is acceptable.
    Exit: At the end of the recommended hold period, or if a stop is hit.
    """
    name = "multi_horizon_hold"

    def __init__(self,
                 max_horizon: int = 3,
                 max_std_multiplier: float = 2.5,
                 min_consistency: float = 0.67,
                 min_sharpe_ratio: float = 0.3,
                 stop_atr_mult: float = 1.5):
        self.max_horizon = max_horizon
        self.max_std_multiplier = max_std_multiplier
        self.min_consistency = min_consistency
        self.min_sharpe_ratio = min_sharpe_ratio
        self.stop_atr_mult = stop_atr_mult

    def _atr(self, history: pd.DataFrame) -> float:
        high = history["high"].values
        low = history["low"].values
        close = history["close"].values
        if len(close) < 2:
            return 0.0
        tr1 = high[-1] - low[-1]
        tr2 = abs(high[-1] - close[-2])
        tr3 = abs(low[-1] - close[-2])
        return float(np.mean([tr1, tr2, tr3]))

    def generate_signal(self, dist, current_price, history, context):
        # We need the multi-horizon stack. The context should contain it,
        # or we compute it here if not present.
        stack = context.get("horizon_stack")
        if stack is None:
            # Compute it on the fly (expensive but works)
            from kairos_horizon import KairosMultiHorizonPredictor
            predictor = KairosMultiHorizonPredictor(
                context.get("predict_fn"),
                max_horizon=self.max_horizon
            )
            stack = predictor.predict(history)

        if not stack.horizons or 1 not in stack.horizons:
            return None

        t1 = stack.horizons[1].stats["close"]
        if t1["std"] == 0:
            return None

        # Direction from T+1
        if t1["mean"] > current_price:
            direction = Direction.LONG
        elif t1["mean"] < current_price:
            direction = Direction.SHORT
        else:
            return None

        # Stop based on ATR
        atr = self._atr(history)
        if direction == Direction.LONG:
            stop = current_price - atr * self.stop_atr_mult
            target = t1["pct_90"]
        else:
            stop = current_price + atr * self.stop_atr_mult
            target = t1["pct_10"]

        # Expected value using T+1 distribution
        ev = stack.horizons[1].expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        # Determine hold period
        rec = stack.hold_recommendation(
            max_std_multiplier=self.max_std_multiplier,
            min_consistency=self.min_consistency,
            min_sharpe_ratio=self.min_sharpe_ratio
        )
        if rec == "flat":
            return None

        # Parse hold days
        hold_days = 1
        if "hold_" in rec:
            try:
                hold_days = int(rec.split("_")[1])
            except (ValueError, IndexError):
                hold_days = 1

        # Size by Kelly on T+1, but reduce if holding longer
        kelly = stack.horizons[1].kelly_fraction(current_price, target, stop)
        hold_penalty = 1.0 / np.sqrt(hold_days)  # Longer hold = more uncertainty
        size = min(kelly * 0.5 * hold_penalty, 1.0)

        # Confidence combines T+1 Sharpe and directional consistency
        t1_sharpe = abs(t1["mean"] - current_price) / t1["std"]
        confidence = min(t1_sharpe * stack.directional_consistency, 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={
                "hold_days": hold_days,
                "horizon_stack": stack,
                "std_decay": stack.std_decay,
                "directional_consistency": stack.directional_consistency,
            }
        )


class ConfidenceDecayFilterStrategy(Strategy):
    """
    Only enters if the confidence decay curve is favorable.
    Favorable = std grows sub-linearly or not at all.
    """
    name = "confidence_decay_filter"

    def __init__(self,
                 max_horizon: int = 3,
                 max_decay_slope: float = 0.8,  # std should not grow faster than 0.8 per horizon
                 min_consistency: float = 0.67):
        self.max_horizon = max_horizon
        self.max_decay_slope = max_decay_slope
        self.min_consistency = min_consistency

    def generate_signal(self, dist, current_price, history, context):
        stack = context.get("horizon_stack")
        if stack is None:
            from kairos_horizon import KairosMultiHorizonPredictor
            predictor = KairosMultiHorizonPredictor(
                context.get("predict_fn"),
                max_horizon=self.max_horizon
            )
            stack = predictor.predict(history)

        if not stack.horizons or 1 not in stack.horizons:
            return None

        # Compute decay slope (linear fit of std vs horizon)
        horizons = sorted(stack.std_decay.keys())
        if len(horizons) >= 2:
            x = np.array(horizons)
            y = np.array([stack.std_decay[h] for h in horizons])
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0

        if slope > self.max_decay_slope:
            return None

        if stack.directional_consistency < self.min_consistency:
            return None

        # If we pass the filter, delegate to a simple directional strategy
        t1 = stack.horizons[1].stats["close"]
        if t1["mean"] > current_price:
            direction = Direction.LONG
            stop = t1["pct_10"]
            target = t1["pct_90"]
        elif t1["mean"] < current_price:
            direction = Direction.SHORT
            stop = t1["pct_90"]
            target = t1["pct_10"]
        else:
            return None

        ev = stack.horizons[1].expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = stack.horizons[1].kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=1.0 - slope / self.max_decay_slope if self.max_decay_slope > 0 else 0.0,
            expected_value=ev,
            metadata={
                "decay_slope": slope,
                "directional_consistency": stack.directional_consistency,
            }
        )


class RollingHorizonStrategy(Strategy):
    """
    Re-evaluates the position each day using fresh predictions.
    If tomorrow's prediction no longer confirms, exit early.
    """
    name = "rolling_horizon"

    def __init__(self,
                 max_horizon: int = 3,
                 min_consistency: float = 0.5,
                 early_exit_threshold: float = 0.3):
        self.max_horizon = max_horizon
        self.min_consistency = min_consistency
        self.early_exit_threshold = early_exit_threshold

    def generate_signal(self, dist, current_price, history, context):
        stack = context.get("horizon_stack")
        if stack is None:
            from kairos_horizon import KairosMultiHorizonPredictor
            predictor = KairosMultiHorizonPredictor(
                context.get("predict_fn"),
                max_horizon=self.max_horizon
            )
            stack = predictor.predict(history)

        if not stack.horizons or 1 not in stack.horizons:
            return None

        t1 = stack.horizons[1].stats["close"]
        if t1["std"] == 0:
            return None

        if t1["mean"] > current_price:
            direction = Direction.LONG
            stop = t1["pct_10"]
            target = t1["pct_90"]
        elif t1["mean"] < current_price:
            direction = Direction.SHORT
            stop = t1["pct_90"]
            target = t1["pct_10"]
        else:
            return None

        ev = stack.horizons[1].expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        # Check if T+2 is still favorable (for early exit planning)
        t2_favorable = False
        if 2 in stack.horizons:
            t2 = stack.horizons[2].stats["close"]
            t2_dir = 1 if t2["mean"] > current_price else (-1 if t2["mean"] < current_price else 0)
            t1_dir = 1 if t1["mean"] > current_price else -1
            t2_favorable = (t2_dir == t1_dir)

        # Reduce size if T+2 is not favorable (expecting early exit)
        kelly = stack.horizons[1].kelly_fraction(current_price, target, stop)
        size = min(kelly * 0.5 * (1.0 if t2_favorable else 0.6), 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=stack.directional_consistency,
            expected_value=ev,
            metadata={
                "t2_favorable": t2_favorable,
                "directional_consistency": stack.directional_consistency,
            }
        )


# =============================================================================
# PATH INTEGRATION STRATEGY
# =============================================================================

class PathIntegrationStrategy(Strategy):
    """
    Analyzes T+1, T+2, T+3 predicted distributions to determine optimal hold period.
    Gets HorizonStack from context["horizon_stack"].

    Key insight: If directional consistency and variance tightening persist across
    multiple horizons, extend the hold period. Otherwise, take the quick 1-day trade.
    """
    name = "path_integration"

    def __init__(self,
                 max_horizon: int = 3,
                 variance_tightening_threshold: float = 0.9,
                 entropy_spike_threshold: float = 1.5):
        self.max_horizon = max_horizon
        self.variance_tightening_threshold = variance_tightening_threshold
        self.entropy_spike_threshold = entropy_spike_threshold

    def _path_quality(self, stack: HorizonStack) -> Tuple[int, float]:
        """
        Analyzes T+1, T+2, T+3 distributions to determine optimal hold period.
        Returns (hold_days, confidence).
        """
        # Get stats for each horizon
        h1_stats = stack.horizons.get(1)
        if h1_stats is None:
            return (1, 0.3)

        h1_stats = h1_stats.stats.get("close")
        if h1_stats is None:
            return (1, 0.3)

        h2_stats = stack.horizons.get(2)
        if h2_stats is None:
            h2_stats = h1_stats
        else:
            h2_stats = h2_stats.stats.get("close", h1_stats)

        h3_stats = stack.horizons.get(3)
        if h3_stats is None:
            h3_stats = h2_stats
        else:
            h3_stats = h3_stats.stats.get("close", h2_stats)

        # Determine direction for each horizon
        base = stack.base_price
        d1 = 1 if h1_stats["mean"] > base else -1
        d2 = 1 if h2_stats["mean"] > base else -1
        d3 = 1 if h3_stats["mean"] > base else -1

        # Check consistency
        consistent = (d1 == d2 == d3)

        # Check variance tightening
        tightening = (
            h2_stats["std"] < h1_stats["std"] * self.variance_tightening_threshold and
            h3_stats["std"] < h2_stats["std"] * self.variance_tightening_threshold
        )

        # Determine hold period and confidence
        if consistent and tightening:
            return (3, 0.9)
        elif consistent:
            return (2, 0.7)
        elif d1 == d2 and d2 != d3:
            return (2, 0.5)
        else:
            return (1, 0.3)

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                       history: pd.DataFrame, context: Dict) -> Optional[Signal]:
        """
        Generates trading signal based on multi-horizon path analysis.
        """
        # Get horizon stack from context
        horizon_stack = context.get("horizon_stack")
        if horizon_stack is None:
            return None

        # Ensure T+1 is available
        h1 = horizon_stack.horizons.get(1)
        if h1 is None:
            return None

        # Determine hold period and path confidence
        hold_days, path_confidence = self._path_quality(horizon_stack)

        # Get T+1 stats
        h1_stats = h1.stats.get("close")
        if h1_stats is None:
            return None

        # Determine direction
        if h1_stats["mean"] > current_price:
            direction = Direction.LONG
            stop = h1_stats["pct_10"]
            target = h1_stats["pct_90"]
        elif h1_stats["mean"] < current_price:
            direction = Direction.SHORT
            stop = h1_stats["pct_90"]
            target = h1_stats["pct_10"]
        else:
            return None

        # Calculate expected value using T+1 distribution
        ev = h1.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        # Calculate size
        kelly = h1.kelly_fraction(current_price, target, stop)
        size = kelly * 0.5 * path_confidence
        size = max(0.01, size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=path_confidence,
            expected_value=ev,
            metadata={
                "hold_days": hold_days,
                "path_confidence": path_confidence,
            }
        )


# =============================================================================
# BACKTEST ENGINE EXTENSION
# =============================================================================

class MultiHorizonBacktestEngine:
    """
    Extended backtester that supports multi-day holds and daily re-evaluation.
    """

    def __init__(self,
                 predictor: KairosMultiHorizonPredictor,
                 fee_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 initial_capital: float = 10000.0):
        self.predictor = predictor
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.initial_capital = initial_capital

        self.trades = []
        self.equity_curve = []
        self.signals = []

        # Active positions with remaining hold days
        self.active_positions = []

    def run(self, df: pd.DataFrame, router, lookback: int = 100) -> Dict:
        capital = self.initial_capital
        prev_capital = capital

        for i in range(lookback, len(df) - 1):
            today = df.iloc[i]
            tomorrow = df.iloc[i + 1]
            date = df.index[i]
            current_price = float(today["close"])

            history = df.iloc[:i + 1]

            # Compute multi-horizon stack
            stack = self.predictor.predict(history)

            # Manage existing positions (daily re-evaluation)
            remaining_positions = []
            for pos in self.active_positions:
                # Check if we should exit today
                exit_price, exit_reason = self._check_exit(pos, tomorrow)
                if exit_price is not None:
                    pnl = self._calculate_pnl(pos, exit_price)
                    capital += pnl
                    self.trades.append({
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "direction": pos["direction"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "size": pos["size"],
                        "pnl": pnl,
                        "strategy_name": pos["strategy_name"],
                        "exit_reason": exit_reason,
                    })
                else:
                    pos["hold_days_remaining"] -= 1
                    if pos["hold_days_remaining"] <= 0:
                        # Force exit at close
                        exit_price = float(tomorrow["close"])
                        pnl = self._calculate_pnl(pos, exit_price)
                        capital += pnl
                        self.trades.append({
                            "entry_date": pos["entry_date"],
                            "exit_date": date,
                            "direction": pos["direction"],
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "size": pos["size"],
                            "pnl": pnl,
                            "strategy_name": pos["strategy_name"],
                            "exit_reason": "hold_expired",
                        })
                    else:
                        remaining_positions.append(pos)

            self.active_positions = remaining_positions

            # Generate new signal
            context = {
                "date": date,
                "current_price": current_price,
                "capital": capital,
                "horizon_stack": stack,
                "predict_fn": self.predictor.predict_fn,
            }
            signal = router.route(stack.horizons[1], current_price, history, context)

            if signal and signal.direction != Direction.FLAT:
                self.signals.append((date, signal))

                # Enter if no conflicting position
                entry_price = float(tomorrow["open"]) * (
                    1.0 + self.slippage_pct * signal.direction.value
                )
                notional = signal.size * capital
                fee = notional * self.fee_pct
                capital -= fee

                hold_days = signal.metadata.get("hold_days", 1) if hasattr(signal, "metadata") else 1

                position = {
                    "direction": signal.direction,
                    "size": notional / entry_price,
                    "entry": entry_price,
                    "stop": signal.stop,
                    "target": signal.target,
                    "entry_date": date,
                    "strategy_name": signal.strategy_name,
                    "hold_days_remaining": hold_days,
                }
                self.active_positions.append(position)

            self.equity_curve.append((date, capital))
            prev_capital = capital

        # Close any remaining positions at the last available close
        if self.active_positions and len(df) > lookback + 1:
            last_close = float(df.iloc[-1]["close"])
            for pos in self.active_positions:
                pnl = self._calculate_pnl(pos, last_close)
                capital += pnl
                self.trades.append({
                    "entry_date": pos["entry_date"],
                    "exit_date": df.index[-1],
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": last_close,
                    "size": pos["size"],
                    "pnl": pnl,
                    "strategy_name": pos["strategy_name"],
                    "exit_reason": "end_of_data",
                })
            self.active_positions = []
            self.equity_curve[-1] = (self.equity_curve[-1][0], capital)

        return self._compute_metrics()

    def _check_exit(self, position, tomorrow):
        direction = position["direction"]
        stop = position["stop"]
        target = position["target"]
        open_p = float(tomorrow["open"])
        high = float(tomorrow["high"])
        low = float(tomorrow["low"])
        close = float(tomorrow["close"])

        if direction == Direction.LONG:
            if open_p <= stop:
                return open_p, "stop_open"
            if open_p >= target:
                return open_p, "target_open"
            if low <= stop:
                return stop, "stop"
            if high >= target:
                return target, "target"
            return None, None
        else:
            if open_p >= stop:
                return open_p, "stop_open"
            if open_p <= target:
                return open_p, "target_open"
            if high >= stop:
                return stop, "stop"
            if low <= target:
                return target, "target"
            return None, None

    def _calculate_pnl(self, position, exit_price):
        if position["direction"] == Direction.LONG:
            gross = (exit_price - position["entry"]) * position["size"]
        else:
            gross = (position["entry"] - exit_price) * position["size"]
        fee = exit_price * position["size"] * self.fee_pct
        return gross - fee

    def _compute_metrics(self):
        if not self.trades:
            return {
                "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0,
                "avg_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            }

        pnls = [t["pnl"] for t in self.trades]
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
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    # Synthetic test
    np.random.seed(42)
    n_samples = 60
    base = 100.0

    def mock_predict(df):
        predictions = []
        for _ in range(n_samples):
            o = base + np.random.normal(0, 0.3)
            l = o - abs(np.random.normal(0.5, 0.2))
            h = o + abs(np.random.normal(1.0, 0.3))
            c = h - abs(np.random.normal(0.2, 0.1))
            predictions.append(pd.DataFrame({
                "open": [o], "high": [h], "low": [l], "close": [c],
                "volume": [1000], "amount": [100000]
            }))
        return predictions

    # Build synthetic history
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    hist = pd.DataFrame({
        "open": [100.0]*10,
        "high": [101.0]*10,
        "low": [99.0]*10,
        "close": [100.5]*10,
        "volume": [1000]*10,
        "amount": [100000]*10,
    }, index=dates)

    predictor = KairosMultiHorizonPredictor(mock_predict, max_horizon=3)
    stack = predictor.predict(hist)
    print(stack)
    print(f"\nRecommendation: {stack.hold_recommendation()}")
    print(f"Confidence curve: {stack.confidence_curve()}")
