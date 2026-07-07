"""
kairos_meta.py
==============
Meta-level strategies for the Kairos framework.

Contains:
  #6 Cross-Asset Sharpe Ranking
  #7 Online Strategy Performance Tracking  
  #9 Kurtosis & Tail Trading

Usage:
    from kairos_meta import (
        MultiAssetKairosPredictor, CrossAssetRankStrategy,
        CrossAssetSpreadStrategy, StrategyPerformanceTracker,
        OnlineWeightedStrategy, KurtosisFilterStrategy,
        TailRiskStrategy, SellPremiumStrategy, BuyWingsStrategy
    )
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple, Any
from collections import defaultdict, deque
import warnings

warnings.filterwarnings("ignore")

# Import base classes (with fallback for standalone use)
try:
    from kairos_backtest import (
        KairosDistribution, KairosPredictor, Direction,
        Signal, Strategy, Trade
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
                        "skew": float(stats.skew(arr)),
                        "kurt": float(stats.kurtosis(arr)),
                        "pct_5": float(np.percentile(arr, 5)),
                        "pct_10": float(np.percentile(arr, 10)),
                        "pct_25": float(np.percentile(arr, 25)),
                        "pct_50": float(np.percentile(arr, 50)),
                        "pct_75": float(np.percentile(arr, 75)),
                        "pct_90": float(np.percentile(arr, 90)),
                        "pct_95": float(np.percentile(arr, 95)),
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


# =============================================================================
# #6 CROSS-ASSET SHARPE RANKING
# =============================================================================

@dataclass
class AssetPrediction:
    """Prediction result for a single asset."""
    symbol: str
    dist: KairosDistribution
    current_price: float
    history: pd.DataFrame


class MultiAssetKairosPredictor:
    """
    Runs Kairos predictions across multiple assets.

    Usage:
        predictor = MultiAssetKairosPredictor(predict_kairos_cloud)
        assets = {
            "BTC-USD": btc_df,
            "ETH-USD": eth_df,
            "SOL-USD": sol_df,
        }
        predictions = predictor.predict_all(assets)
    """

    def __init__(self, predict_fn: Callable[[pd.DataFrame, str], List[pd.DataFrame]],
                 batch_predict_fn: Optional[Callable] = None):
        self.predict_fn = predict_fn
        self.batch_predict_fn = batch_predict_fn

    def predict_all(self, assets: Dict[str, pd.DataFrame]) -> Dict[str, AssetPrediction]:
        if self.batch_predict_fn is not None:
            return self.batch_predict_fn(assets)
        results = {}
        for symbol, df in assets.items():
            current_price = float(df["close"].iloc[-1])
            predictions = self.predict_fn(df, symbol=symbol)
            try:
                from kairos_backtest import distribution_for
                dist = distribution_for(predictions)
            except ImportError:
                dist = KairosDistribution(predictions)
            results[symbol] = AssetPrediction(
                symbol=symbol,
                dist=dist,
                current_price=current_price,
                history=df
            )
        return results


class CrossAssetRankStrategy(Strategy):
    """
    Ranks all assets by predicted Sharpe (mean / std of predicted close).
    Allocates 100% of capital to the top-ranked asset.

    If the top asset's predicted Sharpe is below threshold, goes flat.
    """
    name = "cross_asset_rank"

    def __init__(self, min_sharpe: float = 0.3, top_n: int = 1):
        self.min_sharpe = min_sharpe
        self.top_n = top_n

    def generate_signal(self, dist, current_price, history, context):
        # Expects multi_asset_predictions in context
        predictions = context.get("multi_asset_predictions")
        if not predictions:
            return None

        # Score each asset
        scores = []
        for symbol, pred in predictions.items():
            s = pred.dist.stats["close"]
            if s["std"] == 0:
                continue
            sharpe = abs(s["mean"] - pred.current_price) / s["std"]
            direction = Direction.LONG if s["mean"] > pred.current_price else Direction.SHORT
            scores.append((symbol, sharpe, direction, pred))

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[0]

        if top[1] < self.min_sharpe:
            return None

        symbol, sharpe, direction, pred = top
        s = pred.dist.stats["close"]

        if direction == Direction.LONG:
            stop = pred.dist.stats["low"]["pct_10"]
            target = pred.dist.stats["high"]["pct_90"]
        else:
            stop = pred.dist.stats["high"]["pct_90"]
            target = pred.dist.stats["low"]["pct_10"]

        ev = pred.dist.expected_value(pred.current_price, target, stop)
        if ev <= 0:
            return None

        kelly = pred.dist.kelly_fraction(pred.current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=pred.current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(sharpe, 1.0),
            expected_value=ev,
            metadata={
                "selected_asset": symbol,
                "asset_sharpe": sharpe,
                "all_scores": [(s, sh) for s, sh, _, _ in scores],
            }
        )


class CrossAssetSpreadStrategy(Strategy):
    """
    Pairs trade: long the asset with higher predicted Sharpe,
    short the asset with lower predicted Sharpe (if direction diverges).

    Requires at least 2 assets in context.
    """
    name = "cross_asset_spread"

    def __init__(self, min_sharpe_spread: float = 0.5, min_directional_agreement: float = 0.6):
        self.min_sharpe_spread = min_sharpe_spread
        self.min_directional_agreement = min_directional_agreement

    def generate_signal(self, dist, current_price, history, context):
        predictions = context.get("multi_asset_predictions")
        if not predictions or len(predictions) < 2:
            return None

        # Compute directional agreement and Sharpe for each
        assets = []
        for symbol, pred in predictions.items():
            s = pred.dist.stats["close"]
            if s["std"] == 0:
                continue
            sharpe = (s["mean"] - pred.current_price) / s["std"]
            direction = 1 if s["mean"] > pred.current_price else -1
            assets.append((symbol, sharpe, direction, pred))

        if len(assets) < 2:
            return None

        assets.sort(key=lambda x: x[1], reverse=True)
        top = assets[0]
        bottom = assets[-1]

        # Need clear divergence: top bullish, bottom bearish (or vice versa)
        if top[2] == bottom[2]:
            return None

        sharpe_spread = abs(top[1] - bottom[1])
        if sharpe_spread < self.min_sharpe_spread:
            return None

        # We return a signal for the LONG leg only.
        # The SHORT leg would need a separate signal or the engine
        # would need to support multi-leg positions.
        # For simplicity, we return the long leg and the user can
        # run this strategy twice (once inverted) or use a pairs engine.

        long_asset = top if top[2] == 1 else bottom
        short_asset = bottom if top[2] == 1 else top

        # Return signal for long leg
        pred = long_asset[3]
        s = pred.dist.stats["close"]
        stop = pred.dist.stats["low"]["pct_10"]
        target = pred.dist.stats["high"]["pct_90"]

        ev = pred.dist.expected_value(pred.current_price, target, stop)
        if ev <= 0:
            return None

        kelly = pred.dist.kelly_fraction(pred.current_price, target, stop)
        return Signal(
            direction=Direction.LONG,
            size=min(kelly * 0.5, 1.0),
            entry=pred.current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(long_asset[1]), 1.0),
            expected_value=ev,
            metadata={
                "long_asset": long_asset[0],
                "short_asset": short_asset[0],
                "long_sharpe": long_asset[1],
                "short_sharpe": short_asset[1],
                "spread": sharpe_spread,
            }
        )


class CrossAssetMomentumTransferStrategy(Strategy):
    """
    If BTC predicted tight + bullish, but ETH hasn't moved yet,
    front-run ETH using BTC's foreknowledge.
    """
    name = "cross_asset_momentum_transfer"

    def __init__(self, leader_symbol: str = "BTC-USD", lag_threshold: float = 0.01):
        self.leader_symbol = leader_symbol
        self.lag_threshold = lag_threshold

    def generate_signal(self, dist, current_price, history, context):
        predictions = context.get("multi_asset_predictions")
        if not predictions or self.leader_symbol not in predictions:
            return None

        leader = predictions[self.leader_symbol]
        current_symbol = context.get("current_symbol", "")
        if current_symbol == self.leader_symbol:
            return None

        if current_symbol not in predictions:
            return None

        l_dist = leader.dist.stats["close"]
        c_dist = predictions[current_symbol].dist.stats["close"]
        c_price = predictions[current_symbol].current_price

        # Leader is tight and directional
        leader_cv = l_dist["std"] / l_dist["mean"] if l_dist["mean"] != 0 else 1.0
        if leader_cv > 0.02:
            return None

        leader_move = (l_dist["mean"] - leader.current_price) / leader.current_price
        if abs(leader_move) < 0.005:
            return None

        # Current asset is flat (predicted close near current price)
        current_move = (c_dist["mean"] - c_price) / c_price
        if abs(current_move) > self.lag_threshold:
            return None

        # Trade in leader's direction
        direction = Direction.LONG if leader_move > 0 else Direction.SHORT
        if direction == Direction.LONG:
            stop = predictions[current_symbol].dist.stats["low"]["pct_10"]
            target = predictions[current_symbol].dist.stats["high"]["pct_90"]
        else:
            stop = predictions[current_symbol].dist.stats["high"]["pct_90"]
            target = predictions[current_symbol].dist.stats["low"]["pct_10"]

        ev = predictions[current_symbol].dist.expected_value(c_price, target, stop)
        if ev <= 0:
            return None

        kelly = predictions[current_symbol].dist.kelly_fraction(c_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=c_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(leader_move) / (leader_cv + 0.001),
            expected_value=ev,
            metadata={
                "leader": self.leader_symbol,
                "leader_move": leader_move,
                "follower": current_symbol,
            }
        )


def _factor_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally z-score a factor panel.

    Parameters
    ----------
    panel : DataFrame, index = symbols, columns = factor names.

    Returns
    -------
    DataFrame of the same shape where each column has been z-scored across
    symbols (population std, ddof=0). Columns with zero std map to all zeros.
    """
    z = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for col in panel.columns:
        vals = panel[col].values.astype(float)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        z[col] = (vals - mu) / sd if sd > 0 else np.zeros(len(vals))
    return z


class MultiFactorRankStrategy(Strategy):
    """
    Cross-sectional multi-factor composite rank strategy (design doc §6.4).

    Extends the CrossAssetRankStrategy idea from 1 factor (predicted Sharpe)
    to k factors computed from a trailing returns panel of the universe:

      momentum : 12-month-minus-1-month proxy — sum of returns rows
                 [-252:-21], or the full window minus the last 21 rows when
                 the window is shorter (min 60 rows overall).
      low_vol  : -std(last 60 returns)  (low volatility scores high)
      value    : distance below the window high of the cumulative price
                 index: -(current_index / max_index - 1)  (>= 0; larger =
                 further below high = "cheaper")
      quality  : mean(last 60 returns) / std(last 60 returns)

    Each factor is z-scored cross-sectionally (see `_factor_zscores`), then
    combined into a composite as a weighted mean of z-scores (`weights=None`
    → equal weights).

    Context contract (graceful None when missing):
      context["returns_window"]   : PRIMARY — DataFrame of daily returns,
                                    rows = days, columns = symbols.
      context["universe_history"] : fallback — Dict[symbol, DataFrame] with
                                    a "close" column; returns derived via
                                    pct_change.
      context["symbol"]           : the symbol being evaluated
                                    ("current_symbol" accepted as fallback).

    Trade logic, gated on Kronos agreement:
      current symbol TOP-ranked composite AND Kronos bullish
        (dist close mean > current_price) -> LONG
      current symbol BOTTOM-ranked AND Kronos bearish -> SHORT
      otherwise -> None.

    Bracket: stop = close pct_15, target = close pct_85 (reversed for
    SHORT). Size = min(kelly * 0.5, 1). Confidence = |composite z| squashed
    to (0, 1] via z / (1 + |z|).
    """
    name = "multi_factor_rank"

    FACTOR_NAMES = ("momentum", "low_vol", "value", "quality")
    MIN_ROWS = 60

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights is None:
            weights = {f: 1.0 for f in self.FACTOR_NAMES}
        self.weights = {f: float(weights.get(f, 0.0)) for f in self.FACTOR_NAMES}

    # ------------------------------------------------------------------ #
    def _returns_panel(self, context) -> Optional[pd.DataFrame]:
        """Build the returns panel from context (returns_window primary)."""
        rw = context.get("returns_window")
        if isinstance(rw, pd.DataFrame) and not rw.empty:
            return rw
        uh = context.get("universe_history")
        if not uh:
            return None
        cols = {}
        for symbol, df in uh.items():
            if df is None or "close" not in df:
                continue
            cols[symbol] = (
                df["close"].astype(float).pct_change().dropna().reset_index(drop=True)
            )
        if not cols:
            return None
        return pd.DataFrame(cols).dropna()

    def _compute_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute the raw factor panel (index=symbols, columns=factors)."""
        rows = {}
        for symbol in returns.columns:
            r = returns[symbol].values.astype(float)
            n = len(r)
            # momentum: 12-1 month proxy
            if n >= 252:
                momentum = float(np.sum(r[-252:-21]))
            else:
                momentum = float(np.sum(r[:-21]))
            last60 = r[-60:]
            sd60 = float(np.std(last60))
            low_vol = -sd60
            # value proxy: distance below the window high of the cum index
            cum = np.cumprod(1.0 + r)
            value = -(float(cum[-1]) / float(np.max(cum)) - 1.0)
            quality = float(np.mean(last60)) / sd60 if sd60 > 0 else 0.0
            rows[symbol] = {
                "momentum": momentum, "low_vol": low_vol,
                "value": value, "quality": quality,
            }
        return pd.DataFrame.from_dict(rows, orient="index")[list(self.FACTOR_NAMES)]

    def _composite(self, zscores: pd.DataFrame) -> pd.Series:
        """Weighted mean of factor z-scores per symbol."""
        w = np.array([self.weights[f] for f in self.FACTOR_NAMES], dtype=float)
        if w.sum() == 0:
            w = np.ones(len(w))
        vals = zscores[list(self.FACTOR_NAMES)].values @ w / w.sum()
        return pd.Series(vals, index=zscores.index)

    # ------------------------------------------------------------------ #
    def generate_signal(self, dist, current_price, history, context):
        if not context:
            return None
        symbol = context.get("symbol") or context.get("current_symbol")
        if not symbol:
            return None
        returns = self._returns_panel(context)
        if returns is None or symbol not in returns.columns:
            return None
        if len(returns) < self.MIN_ROWS:
            return None

        factors = self._compute_factors(returns)
        zscores = _factor_zscores(factors)
        composite = self._composite(zscores)

        top_symbol = composite.idxmax()
        bottom_symbol = composite.idxmin()

        s = dist.stats.get("close", {})
        mean = s.get("mean")
        if mean is None or "pct_15" not in s or "pct_85" not in s:
            return None
        bullish = mean > current_price

        if symbol == top_symbol and bullish:
            direction = Direction.LONG
            stop, target = s["pct_15"], s["pct_85"]
        elif symbol == bottom_symbol and not bullish:
            direction = Direction.SHORT
            stop, target = s["pct_85"], s["pct_15"]
        else:
            return None

        z = float(composite[symbol])
        ev = dist.expected_value(current_price, target, stop)
        kelly = dist.kelly_fraction(current_price, target, stop)
        return Signal(
            direction=direction,
            size=min(kelly * 0.5, 1.0),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=abs(z) / (1.0 + abs(z)),
            expected_value=ev,
            metadata={
                "symbol": symbol,
                "composite_z": z,
                "rank_top": top_symbol,
                "rank_bottom": bottom_symbol,
                "composite_scores": composite.to_dict(),
                "factor_zscores": zscores.loc[symbol].to_dict(),
                "weights": dict(self.weights),
            }
        )


# =============================================================================
# #7 ONLINE STRATEGY PERFORMANCE TRACKING
# =============================================================================

@dataclass
class StrategyPerformance:
    """Rolling performance metrics for a single strategy."""
    strategy_name: str
    returns: deque = field(default_factory=lambda: deque(maxlen=30))
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    @property
    def sharpe(self) -> float:
        if len(self.returns) < 5:
            return 0.0
        arr = np.array(list(self.returns))
        if np.std(arr) == 0:
            return 0.0
        return float(np.mean(arr) / np.std(arr) * np.sqrt(252))

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        wins = sum(r for r in self.returns if r > 0)
        losses = sum(abs(r) for r in self.returns if r < 0)
        return wins / losses if losses > 0 else float("inf")

    def add_trade(self, pnl: float, entry_price: float, exit_price: float):
        """Add a completed trade."""
        ret = pnl / (entry_price + 0.001)  # approximate return
        self.returns.append(ret)
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1


class StrategyPerformanceTracker:
    """
    Tracks rolling performance for all strategies.
    Singleton-like object that persists across backtest bars.
    """

    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        self.performances: Dict[str, StrategyPerformance] = {}

    def record_trade(self, strategy_name: str, pnl: float, entry_price: float, exit_price: float):
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name,
                returns=deque(maxlen=self.lookback_window)
            )
        self.performances[strategy_name].add_trade(pnl, entry_price, exit_price)

    def get_sharpe(self, strategy_name: str) -> float:
        if strategy_name not in self.performances:
            return 0.0
        return self.performances[strategy_name].sharpe

    def get_weight(self, strategy_name: str, temperature: float = 1.0) -> float:
        """
        Returns a softmax weight based on recent Sharpe.
        Higher temperature = more equal weighting.
        Lower temperature = more concentrated on top performers.
        """
        if strategy_name not in self.performances:
            return 1.0 / len(self.performances) if self.performances else 0.0

        sharpe = self.performances[strategy_name].sharpe
        # Shift to positive for softmax
        sharpe = max(sharpe, -2.0)

        all_sharpes = [p.sharpe for p in self.performances.values()]
        all_sharpes = [max(s, -2.0) for s in all_sharpes]

        if not all_sharpes or temperature == 0:
            return 1.0 / len(all_sharpes) if all_sharpes else 0.0

        exp_scores = [np.exp(s / temperature) for s in all_sharpes]
        total = sum(exp_scores)
        idx = list(self.performances.keys()).index(strategy_name)
        return exp_scores[idx] / total if total > 0 else 0.0

    def get_best_strategy(self, min_trades: int = 5) -> Optional[str]:
        """Returns the strategy with the highest Sharpe that has enough trades."""
        candidates = [
            (name, perf) for name, perf in self.performances.items()
            if (perf.wins + perf.losses) >= min_trades
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1].sharpe)[0]

    def get_all_rankings(self) -> List[Tuple[str, float]]:
        """Returns all strategies ranked by Sharpe."""
        ranked = [(name, perf.sharpe) for name, perf in self.performances.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class OnlineWeightedStrategy(Strategy):
    """
    Meta-strategy that runs multiple base strategies and weights
    their signals by recent realized Sharpe.

    Uses a StrategyPerformanceTracker to maintain state.
    """
    name = "online_weighted"

    def __init__(self, base_strategies: List[Strategy],
                 tracker: Optional[StrategyPerformanceTracker] = None,
                 temperature: float = 0.5,
                 min_weight_threshold: float = 0.05):
        self.base_strategies = base_strategies
        self.tracker = tracker or StrategyPerformanceTracker()
        self.temperature = temperature
        self.min_weight_threshold = min_weight_threshold

    def generate_signal(self, dist, current_price, history, context):
        signals = []
        for strat in self.base_strategies:
            sig = strat.generate_signal(dist, current_price, history, context)
            if sig is None:
                continue

            weight = self.tracker.get_weight(strat.name, self.temperature)
            if weight < self.min_weight_threshold:
                continue

            sig.size *= weight
            sig.metadata["strategy_weight"] = weight
            sig.metadata["strategy_sharpe"] = self.tracker.get_sharpe(strat.name)
            signals.append(sig)

        if not signals:
            return None

        # Pick the highest weighted-EV signal
        best = max(signals, key=lambda s: s.expected_value * s.metadata.get("strategy_weight", 1.0))
        best.strategy_name = self.name
        return best


class ThompsonSamplingStrategy(Strategy):
    """
    Uses Thompson Sampling to select strategies.
    Models each strategy's Sharpe as a Beta distribution
    and samples from it.
    """
    name = "thompson_sampling"

    def __init__(self, base_strategies: List[Strategy],
                 tracker: Optional[StrategyPerformanceTracker] = None):
        self.base_strategies = base_strategies
        self.tracker = tracker or StrategyPerformanceTracker()

    def _sample_sharpe(self, strategy_name: str) -> float:
        if strategy_name not in self.tracker.performances:
            return 0.0
        perf = self.tracker.performances[strategy_name]
        if perf.wins + perf.losses < 3:
            return 0.5  # Prior: moderate optimism

        # Approximate Beta: alpha = wins + 1, beta = losses + 1
        # Then scale to Sharpe range [-2, 2]
        alpha = perf.wins + 1
        beta = perf.losses + 1
        sample = np.random.beta(alpha, beta)
        return (sample - 0.5) * 4.0  # Map [0,1] to [-2, 2]

    def generate_signal(self, dist, current_price, history, context):
        # Sample a Sharpe for each strategy
        samples = []
        for strat in self.base_strategies:
            sampled_sharpe = self._sample_sharpe(strat.name)
            sig = strat.generate_signal(dist, current_price, history, context)
            if sig is None:
                continue
            sig.metadata["sampled_sharpe"] = sampled_sharpe
            samples.append((sig, sampled_sharpe))

        if not samples:
            return None

        # Pick the strategy with the highest sampled Sharpe
        best_sig, best_sharpe = max(samples, key=lambda x: x[1])
        best_sig.strategy_name = self.name
        best_sig.confidence = min(best_sharpe / 2.0, 1.0)  # Normalize
        return best_sig


class RegimeSwitchingStrategy(Strategy):
    """
    Switches between strategy sets based on detected regime.
    Tracks which strategy set performs best in each regime.
    """
    name = "regime_switching"

    def __init__(self,
                 regime_strategies: Dict[str, List[Strategy]],
                 tracker: Optional[StrategyPerformanceTracker] = None):
        self.regime_strategies = regime_strategies
        self.tracker = tracker or StrategyPerformanceTracker()
        self.regime_history: deque = deque(maxlen=100)

    def _detect_regime(self, dist: KairosDistribution, current_price: float) -> str:
        s = dist.stats["close"]
        cv = s["std"] / abs(s["mean"]) if s["mean"] != 0 else 1.0
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price

        if cv < 0.02 and pred_range < 0.03:
            return "range"
        elif cv > 0.04 or pred_range > 0.05:
            return "trend"
        return "uncertain"

    def generate_signal(self, dist, current_price, history, context):
        regime = self._detect_regime(dist, current_price)
        strategies = self.regime_strategies.get(regime, [])
        if not strategies:
            return None

        # Weight by recent Sharpe within this regime
        signals = []
        for strat in strategies:
            sig = strat.generate_signal(dist, current_price, history, context)
            if sig is None:
                continue
            weight = self.tracker.get_weight(strat.name, temperature=0.5)
            sig.size *= weight
            sig.metadata["regime"] = regime
            sig.metadata["regime_weight"] = weight
            signals.append(sig)

        if not signals:
            return None

        best = max(signals, key=lambda s: s.expected_value)
        best.strategy_name = self.name
        return best


# =============================================================================
# #9 KURTOSIS & TAIL TRADING
# =============================================================================

class KurtosisFilterStrategy(Strategy):
    """
    Wraps a base strategy with a kurtosis filter.

    High kurtosis (> 3) = fat tails = avoid or buy wings.
    Low kurtosis (< 0) = thin tails = safe to sell premium or take directional.
    """
    name = "kurtosis_filter"

    def __init__(self, base_strategy: Strategy,
                 max_kurtosis: float = 3.0,
                 action: str = "block"):
        """
        action: "block" = no trade if kurtosis > max
                "reduce" = halve size if kurtosis > max
                "invert" = fade the move if kurtosis > max
        """
        self.base_strategy = base_strategy
        self.max_kurtosis = max_kurtosis
        self.action = action

    def generate_signal(self, dist, current_price, history, context):
        kurt = dist.stats["close"].get("kurt", 0.0)

        if kurt > self.max_kurtosis:
            if self.action == "block":
                return None
            elif self.action == "reduce":
                sig = self.base_strategy.generate_signal(dist, current_price, history, context)
                if sig:
                    sig.size *= 0.5
                    sig.metadata["kurtosis_penalty"] = 0.5
                return sig
            elif self.action == "invert":
                sig = self.base_strategy.generate_signal(dist, current_price, history, context)
                if sig:
                    sig.direction = Direction.SHORT if sig.direction == Direction.LONG else Direction.LONG
                    sig.strategy_name = f"{self.name}_inverted"
                    sig.metadata["kurtosis_inverted"] = True
                return sig

        return self.base_strategy.generate_signal(dist, current_price, history, context)


class TailRiskStrategy(Strategy):
    """
    Buys protective options / hedges when predicted distribution
    shows fat tails (high kurtosis or wide percentiles).

    For spot/futures: reduces position size or buys a straddle proxy.
    For this framework: returns a FLAT signal with metadata indicating hedge.
    """
    name = "tail_risk"

    def __init__(self, kurtosis_threshold: float = 2.0,
                 tail_width_threshold: float = 0.05):
        self.kurtosis_threshold = kurtosis_threshold
        self.tail_width_threshold = tail_width_threshold

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        kurt = s.get("kurt", 0.0)
        tail_width = (s["pct_95"] - s["pct_5"]) / current_price

        if kurt < self.kurtosis_threshold and tail_width < self.tail_width_threshold:
            return None

        # Signal to hedge: reduce exposure or buy wings
        return Signal(
            direction=Direction.FLAT,
            size=0.0,
            entry=current_price,
            stop=0.0,
            target=0.0,
            strategy_name=self.name,
            confidence=min(kurt / self.kurtosis_threshold, 1.0),
            expected_value=0.0,
            metadata={
                "action": "hedge",
                "kurtosis": kurt,
                "tail_width": tail_width,
                "recommended": "buy_straddle_or_reduce_size",
            }
        )


class SellPremiumStrategy(Strategy):
    """
    Sells straddles / strangles when predicted kurtosis is low
    (thin tails) and predicted realized vol is below implied vol.

    Returns a FLAT signal with metadata for options execution.
    """
    name = "sell_premium"

    def __init__(self, max_kurtosis: float = 0.5,
                 min_iv_premium: float = 0.2,
                 min_range_pct: float = 0.01):
        self.max_kurtosis = max_kurtosis
        self.min_iv_premium = min_iv_premium
        self.min_range_pct = min_range_pct

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        kurt = s.get("kurt", 0.0)
        pred_vol = s["std"] / current_price
        pred_range = (s["pct_90"] - s["pct_10"]) / current_price

        # Get implied vol from context if available
        iv = context.get("implied_vol", pred_vol * 1.5)  # Default: assume 50% premium
        iv_premium = (iv - pred_vol) / iv if iv > 0 else 0.0

        if kurt > self.max_kurtosis:
            return None
        if iv_premium < self.min_iv_premium:
            return None
        if pred_range < self.min_range_pct:
            return None

        return Signal(
            direction=Direction.FLAT,
            size=0.0,
            entry=current_price,
            stop=0.0,
            target=0.0,
            strategy_name=self.name,
            confidence=min(iv_premium, 1.0),
            expected_value=0.0,
            metadata={
                "action": "sell_straddle",
                "strike_low": s["pct_10"],
                "strike_high": s["pct_90"],
                "pred_vol": pred_vol,
                "implied_vol": iv,
                "iv_premium": iv_premium,
                "kurtosis": kurt,
            }
        )


class BuyWingsStrategy(Strategy):
    """
    Buys out-of-the-money options when predicted kurtosis is high
    (fat tails) and the market hasn't priced them.

    Returns a FLAT signal with metadata for options execution.
    """
    name = "buy_wings"

    def __init__(self, min_kurtosis: float = 2.0,
                 min_tail_width_pct: float = 0.04,
                 max_iv: float = 0.5):
        self.min_kurtosis = min_kurtosis
        self.min_tail_width_pct = min_tail_width_pct
        self.max_iv = max_iv

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        kurt = s.get("kurt", 0.0)
        tail_width = (s["pct_95"] - s["pct_5"]) / current_price
        pred_vol = s["std"] / current_price

        iv = context.get("implied_vol", pred_vol)

        if kurt < self.min_kurtosis:
            return None
        if tail_width < self.min_tail_width_pct:
            return None
        if iv > self.max_iv:
            return None  # Too expensive

        return Signal(
            direction=Direction.FLAT,
            size=0.0,
            entry=current_price,
            stop=0.0,
            target=0.0,
            strategy_name=self.name,
            confidence=min(kurt / self.min_kurtosis, 1.0),
            expected_value=0.0,
            metadata={
                "action": "buy_wings",
                "put_strike": s["pct_5"],
                "call_strike": s["pct_95"],
                "pred_vol": pred_vol,
                "implied_vol": iv,
                "kurtosis": kurt,
                "tail_width": tail_width,
            }
        )


class TailAsymmetryStrategy(Strategy):
    """
    Trades the asymmetry between left and right tails.

    Compares 5th percentile distance to 95th percentile distance.
    If left tail is fatter than right tail (more downside risk),
    buy protective puts or reduce long exposure.
    If right tail is fatter, buy calls or increase long exposure.
    """
    name = "tail_asymmetry"

    def __init__(self, min_asymmetry_ratio: float = 1.5):
        self.min_asymmetry_ratio = min_asymmetry_ratio

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        left_tail = current_price - s["pct_5"]
        right_tail = s["pct_95"] - current_price

        if left_tail <= 0 or right_tail <= 0:
            return None

        ratio = left_tail / right_tail
        inv_ratio = right_tail / left_tail

        if ratio > self.min_asymmetry_ratio:
            # Left tail fatter: more downside risk
            direction = Direction.SHORT
            stop = s["pct_95"]
            target = s["pct_5"]
            confidence = min(ratio / self.min_asymmetry_ratio, 1.0)
            asymmetry = "left_fat"
        elif inv_ratio > self.min_asymmetry_ratio:
            # Right tail fatter: more upside potential
            direction = Direction.LONG
            stop = s["pct_5"]
            target = s["pct_95"]
            confidence = min(inv_ratio / self.min_asymmetry_ratio, 1.0)
            asymmetry = "right_fat"
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
            confidence=confidence,
            expected_value=ev,
            metadata={
                "asymmetry": asymmetry,
                "left_tail": left_tail,
                "right_tail": right_tail,
                "ratio": ratio,
            }
        )


class PercentileTailStrategy(Strategy):
    """
    Sells options at the 5th and 95th percentiles.
    The model tells you where the tails actually are.
    """
    name = "percentile_tail"

    def __init__(self, min_distance_pct: float = 0.02):
        self.min_distance_pct = min_distance_pct

    def generate_signal(self, dist, current_price, history, context):
        s = dist.stats["close"]
        put_strike = s["pct_5"]
        call_strike = s["pct_95"]

        put_dist = (current_price - put_strike) / current_price
        call_dist = (call_strike - current_price) / current_price

        if put_dist < self.min_distance_pct or call_dist < self.min_distance_pct:
            return None

        return Signal(
            direction=Direction.FLAT,
            size=0.0,
            entry=current_price,
            stop=0.0,
            target=0.0,
            strategy_name=self.name,
            confidence=min(put_dist, call_dist) / self.min_distance_pct,
            expected_value=0.0,
            metadata={
                "action": "sell_strangle",
                "put_strike": put_strike,
                "call_strike": call_strike,
                "put_dist": put_dist,
                "call_dist": call_dist,
            }
        )


# =============================================================================
# REGIME CLUSTER STRATEGY (KNN-based strategy selector)
# =============================================================================

class RegimeClusterStrategy(Strategy):
    """
    KNN-based meta-strategy that selects the best-performing base strategy
    based on similarity to recent market regimes.

    Records market features and strategy performance in a buffer.
    When a new signal is needed, finds k-nearest neighbors in feature space
    and selects the strategy that performed best in similar regimes.
    """
    name = "regime_cluster"

    def __init__(self, base_strategies: List[Strategy],
                 feature_buffer_size: int = 100,
                 k_neighbors: int = 5,
                 distance_threshold: float = 0.5,
                 fallback_strategy: Optional[Strategy] = None):
        self.base_strategies = base_strategies
        self.feature_buffer_size = feature_buffer_size
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.fallback_strategy = fallback_strategy
        self.feature_buffer: deque = deque(maxlen=feature_buffer_size)

    def record_trade(self, features: List[float], strategy_name: str, pnl: float):
        """
        Record a completed trade with its market features and outcome.

        Args:
            features: 5D feature vector from market regime
            strategy_name: name of the strategy that generated the signal
            pnl: profit/loss from the trade
        """
        self.feature_buffer.append((features, strategy_name, pnl))

    def _extract_features(self, dist: KairosDistribution, current_price: float) -> List[float]:
        """
        Extract 5-dimensional feature vector from distribution and current price.

        Features:
        0. Coefficient of variation (volatility / mean)
        1. Skewness
        2. Percentile range (pct_90 - pct_10) / current_price
        3. Entropy
        4. Trend direction (1.0 if mean > price, -1.0 otherwise)
        """
        s = dist.stats.get("close", {})
        cv = dist.coefficient_of_variation("close")
        skew = s.get("skew", 0.0)
        pct_range = (s.get("pct_90", 0) - s.get("pct_10", 0)) / current_price if current_price > 0 else 0.0
        ent = dist.entropy("close")
        trend = 1.0 if s.get("mean", 0) > current_price else -1.0

        return [cv, skew, pct_range, ent, trend]

    def _normalize_features(self, features: List[float], buffer: deque) -> List[float]:
        """
        Normalize features using min-max scaling based on buffer statistics.
        Handles edge case where all values are identical.
        """
        if not buffer:
            return [0.5] * len(features)

        normalized = []
        for i in range(len(features)):
            vals = [entry[0][i] for entry in buffer]
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                normalized.append(0.5)
            else:
                norm_val = (features[i] - min_v) / (max_v - min_v)
                normalized.append(max(0.0, min(1.0, norm_val)))

        return normalized

    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        """Compute Euclidean distance between two feature vectors."""
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2))))

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        # Extract current market features
        features = self._extract_features(dist, current_price)

        # If buffer is too small, use fallback strategy if available
        if len(self.feature_buffer) < self.k_neighbors:
            if self.fallback_strategy is not None:
                return self.fallback_strategy.generate_signal(dist, current_price, history, context)
            return None

        # Normalize features (current point against buffer)
        normalized_features = self._normalize_features(features, self.feature_buffer)

        # Compute distances to all buffer entries
        distances = []
        for buf_entry in self.feature_buffer:
            buf_features_normalized = self._normalize_features(buf_entry[0], self.feature_buffer)
            dist_val = self._euclidean_distance(normalized_features, buf_features_normalized)
            distances.append((dist_val, buf_entry))

        # Sort by distance and select k-nearest
        distances.sort(key=lambda x: x[0])
        nearest_k = distances[:self.k_neighbors]

        # If minimum distance is too large, use fallback or return None
        if nearest_k[0][0] > self.distance_threshold:
            if self.fallback_strategy is not None:
                return self.fallback_strategy.generate_signal(dist, current_price, history, context)
            return None

        # Group by strategy name and compute mean pnl
        strategy_pnls: Dict[str, List[float]] = defaultdict(list)
        for _, (_, strategy_name, pnl) in nearest_k:
            strategy_pnls[strategy_name].append(pnl)

        # Pick strategy with highest mean pnl
        best_strategy_name = max(
            strategy_pnls.keys(),
            key=lambda s: np.mean(strategy_pnls[s])
        )

        # Find and run the best strategy
        best_strategy = None
        for strat in self.base_strategies:
            if strat.name == best_strategy_name:
                best_strategy = strat
                break

        if best_strategy is None:
            if self.fallback_strategy is not None:
                return self.fallback_strategy.generate_signal(dist, current_price, history, context)
            return None

        # Generate signal from best strategy
        sig = best_strategy.generate_signal(dist, current_price, history, context)
        if sig:
            sig.metadata["regime_cluster_selected_strategy"] = best_strategy_name
            sig.metadata["regime_cluster_mean_pnl"] = float(np.mean(strategy_pnls[best_strategy_name]))
            sig.metadata["regime_cluster_nearest_k"] = len(nearest_k)

        return sig


# =============================================================================
# MONTE CARLO SCENARIO STRATEGY
# =============================================================================

class MonteCarloScenarioStrategy(Strategy):
    """
    Evaluates base strategies against Monte Carlo synthetic close scenarios.

    For each base strategy, simulates the signal against n_scenarios synthetic
    closes drawn from N(mean, std) to compute expected PnL and Sharpe ratio.
    Selects the strategy with the highest expected value (or Sharpe ratio,
    depending on selection_metric) and returns its REAL signal (on actual dist).
    """
    name = "monte_carlo_scenario"

    def __init__(self, base_strategies: List[Strategy],
                 n_scenarios: int = 1000,
                 selection_metric: str = "expected_pnl"):
        """
        Args:
            base_strategies: List of strategies to evaluate
            n_scenarios: Number of synthetic closes to generate
            selection_metric: "expected_pnl" or "sharpe" for strategy selection
        """
        self.base_strategies = base_strategies
        self.n_scenarios = n_scenarios
        self.selection_metric = selection_metric

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        s = dist.stats.get("close", {})
        mean = s.get("mean", current_price)
        std = s.get("std", 0.0)

        # If no volatility, use fallback (first strategy)
        if std == 0:
            if self.base_strategies:
                return self.base_strategies[0].generate_signal(dist, current_price, history, context)
            return None

        # Generate synthetic closes from normal distribution
        synthetic = np.random.normal(mean, std, self.n_scenarios)

        # Evaluate each strategy against synthetic scenarios
        strategy_scores = []
        strategy_signals = {}

        for strategy in self.base_strategies:
            # Generate signal from this strategy on real distribution
            sig = strategy.generate_signal(dist, current_price, history, context)

            if sig is None or sig.direction == Direction.FLAT:
                expected_pnl = 0.0
                sharpe = 0.0
            else:
                # Simulate PnL across synthetic scenarios
                direction_sign = sig.direction.value
                pnls = direction_sign * (synthetic - sig.entry) * sig.size
                expected_pnl = float(np.mean(pnls))
                std_pnl = float(np.std(pnls))
                sharpe = expected_pnl / std_pnl if std_pnl > 0 else 0.0

            strategy_signals[strategy.name] = sig

            # Select metric for ranking
            if self.selection_metric == "sharpe":
                score = sharpe
            else:  # expected_pnl
                score = expected_pnl

            strategy_scores.append((strategy.name, score, expected_pnl, sharpe))

        if not strategy_scores:
            return None

        # Pick strategy with highest score
        best_strategy_name, best_score, best_expected_pnl, best_sharpe = max(
            strategy_scores,
            key=lambda x: x[1]
        )

        # Return the best strategy's real signal
        best_sig = strategy_signals[best_strategy_name]
        if best_sig:
            best_sig.metadata["monte_carlo_selected_strategy"] = best_strategy_name
            best_sig.metadata["monte_carlo_expected_pnl"] = best_expected_pnl
            best_sig.metadata["monte_carlo_sharpe"] = best_sharpe
            best_sig.metadata["monte_carlo_n_scenarios"] = self.n_scenarios

        return best_sig


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    print("kairos_meta.py loaded successfully.")
    print("Cross-Asset:")
    print("  MultiAssetKairosPredictor, CrossAssetRankStrategy")
    print("  CrossAssetSpreadStrategy, CrossAssetMomentumTransferStrategy")
    print("Online Tracking:")
    print("  StrategyPerformanceTracker, OnlineWeightedStrategy")
    print("  ThompsonSamplingStrategy, RegimeSwitchingStrategy")
    print("Kurtosis / Tail:")
    print("  KurtosisFilterStrategy, TailRiskStrategy, SellPremiumStrategy")
    print("  BuyWingsStrategy, TailAsymmetryStrategy, PercentileTailStrategy")
    print("KNN Regime & Monte Carlo:")
    print("  RegimeClusterStrategy, MonteCarloScenarioStrategy")
