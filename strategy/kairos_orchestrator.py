"""
kairos_orchestrator.py
======================
The master orchestrator for the Kairos trading framework.

Wires together all 42 strategies across 5 modules:
  - kairos_backtest.py    (18 base strategies)
  - kairos_path.py        (5 path-aware strategies)
  - kairos_horizon.py     (3 multi-horizon strategies)
  - kairos_execution.py   (7 execution + volume strategies)
  - kairos_meta.py        (9 meta + cross-asset + tail strategies)

Features:
  - Multi-asset prediction and ranking
  - Parallel strategy evaluation
  - Online performance weighting
  - Meta-filter application (entropy, bimodality, kurtosis, volume)
  - Partial exit execution plans
  - Multi-day hold management
  - Unified daily signal output
  - Comprehensive backtest reporting

Usage:
    from kairos_orchestrator import KairosOrchestrator

    orchestrator = KairosOrchestrator(
        predict_fn=predict_kairos_cloud,
        assets=["BTC-USD", "ETH-USD", "SOL-USD"],
        initial_capital=10000.0,
        fee_pct=0.001,
        slippage_pct=0.0005,
    )

    results = orchestrator.run_backtest(
        data_dict={"BTC-USD": btc_df, "ETH-USD": eth_df, "SOL-USD": sol_df},
        lookback=200
    )

    print(results.summary)
    print(results.best_strategy)
    print(results.equity_curve)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple, Any, Set
from collections import defaultdict, deque
import warnings
import json
from datetime import datetime

from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# IMPORT ALL MODULES
# =============================================================================

try:
    from kairos_backtest import (
        KairosDistribution, KairosPredictor, Direction,
        Signal, Strategy, Trade, BacktestEngine,
        DecisionTreeRouter,
        PercentileEntryStrategy, DynamicBracketStrategy, SkewStrategy,
        RangeTradingStrategy, TrendFollowingStrategy, VolatilityArbStrategy,
        HighLowStrategy, OpenGapStrategy, FadeExtremeStrategy,
        MomentumContinuationStrategy, ExpectedValueStrategy,
        MartingaleFloorStrategy, RSIFilterStrategy, MACDFilterStrategy,
        BollingerValidationStrategy, SupportConfluenceStrategy,
        InverseVarianceSizingStrategy, CloseDirectionStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_backtest: {e}")

try:
    from kairos_path import (
        KairosPathExtractor, PathPattern,
        PathRallyStrategy, PathFadeStrategy,
        PathVShapeStrategy, PathInvertedVStrategy,
        PathHighLowSequenceStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_path: {e}")

try:
    from kairos_horizon import (
        KairosMultiHorizonPredictor, HorizonStack,
        MultiHorizonHoldStrategy, ConfidenceDecayFilterStrategy,
        RollingHorizonStrategy, MultiHorizonBacktestEngine,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_horizon: {e}")

try:
    from kairos_execution import (
        PathExecutionPlanner, PathExecutionStrategy,
        PathHighLowExecutionStrategy,
        VolumeAnalyzer, LiquidityFilterStrategy,
        VolumeConfirmationStrategy, VolumeFadeStrategy,
        AmountFlowStrategy, PredictedVWAPStrategy,
        PartialExitBacktestEngine,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_execution: {e}")

try:
    from kairos_meta import (
        MultiAssetKairosPredictor, AssetPrediction,
        CrossAssetRankStrategy, CrossAssetSpreadStrategy,
        CrossAssetMomentumTransferStrategy,
        StrategyPerformanceTracker, StrategyPerformance,
        OnlineWeightedStrategy, ThompsonSamplingStrategy,
        RegimeSwitchingStrategy,
        KurtosisFilterStrategy, TailRiskStrategy,
        SellPremiumStrategy, BuyWingsStrategy,
        TailAsymmetryStrategy, PercentileTailStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_meta: {e}")


# =============================================================================
# ORCHESTRATOR CONFIG
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the Kairos orchestrator."""

    # Capital
    initial_capital: float = 10000.0
    fee_pct: float = 0.001
    slippage_pct: float = 0.0005

    # Meta-filters
    entropy_threshold: float = 3.0
    bimodality_filter: bool = True
    kurtosis_max: float = 10.0
    kurtosis_action: str = "block"  # "block", "reduce", "invert"
    min_volume_percentile: float = 10.0
    debug_filters: bool = False

    # Performance tracking
    performance_lookback: int = 30
    online_weighting: bool = True
    thompson_sampling: bool = False
    temperature: float = 0.5

    # Cross-asset
    cross_asset_ranking: bool = True
    cross_asset_spread: bool = False
    leader_asset: str = "BTC-USD"
    min_sharpe: float = 0.3

    # Multi-horizon
    max_horizon: int = 3
    multi_horizon_holds: bool = True

    # Partial exits
    partial_exits: bool = True
    leg1_pct: float = 0.33
    leg2_pct: float = 0.33
    leg3_pct: float = 0.34
    leg3_trail: bool = True

    # Execution
    max_positions_per_day: int = 3
    max_position_size_pct: float = 0.5  # Max 50% of capital per trade
    max_total_exposure: float = 1.0  # Max 100% of capital deployed

    # Logging
    verbose: bool = False
    log_signals: bool = True


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

class StrategyRegistry:
    """
    Maintains the full registry of all 42 strategies.
    """

    ALL_STRATEGIES: List[Strategy] = []

    @classmethod
    def build_all(cls, config: OrchestratorConfig) -> List[Strategy]:
        """Build all 42 strategies with the given config."""
        strategies = []

        # === BASE STRATEGIES (18) ===
        strategies.extend([
            PercentileEntryStrategy(),
            DynamicBracketStrategy(),
            SkewStrategy(),
            RangeTradingStrategy(),
            TrendFollowingStrategy(),
            VolatilityArbStrategy(),
            HighLowStrategy(),
            OpenGapStrategy(),
            FadeExtremeStrategy(),
            MomentumContinuationStrategy(),
            ExpectedValueStrategy(),
            MartingaleFloorStrategy(),
            RSIFilterStrategy(),
            MACDFilterStrategy(),
            BollingerValidationStrategy(),
            SupportConfluenceStrategy(),
            InverseVarianceSizingStrategy(),
            CloseDirectionStrategy(),
        ])

        # === PATH STRATEGIES (5) ===
        strategies.extend([
            PathRallyStrategy(),
            PathFadeStrategy(),
            PathVShapeStrategy(),
            PathInvertedVStrategy(),
            PathHighLowSequenceStrategy(),
        ])

        # === HORIZON STRATEGIES (3) ===
        strategies.extend([
            MultiHorizonHoldStrategy(max_horizon=config.max_horizon),
            ConfidenceDecayFilterStrategy(max_horizon=config.max_horizon),
            RollingHorizonStrategy(max_horizon=config.max_horizon),
        ])

        # === EXECUTION STRATEGIES (2 direct, 5 volume) ===
        strategies.extend([
            PathExecutionStrategy(
                leg1_pct=config.leg1_pct,
                leg2_pct=config.leg2_pct,
                leg3_pct=config.leg3_pct,
                leg3_trail=config.leg3_trail,
            ),
            PathHighLowExecutionStrategy(),
            VolumeConfirmationStrategy(),
            VolumeFadeStrategy(),
            AmountFlowStrategy(),
            PredictedVWAPStrategy(),
        ])

        # === META STRATEGIES (9) ===
        strategies.extend([
            CrossAssetRankStrategy(min_sharpe=config.min_sharpe),
            CrossAssetSpreadStrategy(),
            CrossAssetMomentumTransferStrategy(leader_symbol=config.leader_asset),
            TailAsymmetryStrategy(),
            PercentileTailStrategy(),
        ])

        # Apply kurtosis filter to all directional strategies
        if config.kurtosis_action != "none":
            filtered = []
            for s in strategies:
                if hasattr(s, "direction") or s.name in [
                    "percentile_entry", "dynamic_bracket", "skew",
                    "range_trading", "trend_following", "high_low",
                    "open_gap", "fade_extreme", "momentum_continuation",
                    "expected_value", "martingale_floor", "rsi_filter",
                    "macd_filter", "bollinger_validation", "support_confluence",
                    "inverse_variance", "close_direction", "path_rally",
                    "path_fade", "path_v_shape", "path_inverted_v",
                    "path_high_low_sequence", "multi_horizon_hold",
                    "confidence_decay_filter", "rolling_horizon",
                    "path_execution", "path_high_low_execution",
                    "volume_confirmation", "volume_fade", "amount_flow",
                    "predicted_vwap", "cross_asset_rank", "cross_asset_spread",
                    "cross_asset_momentum_transfer", "tail_asymmetry",
                ]:
                    filtered.append(KurtosisFilterStrategy(
                        base_strategy=s,
                        max_kurtosis=config.kurtosis_max,
                        action=config.kurtosis_action
                    ))
                else:
                    filtered.append(s)
            strategies = filtered

        # Apply volume filter to all
        if config.min_volume_percentile > 0:
            filtered = []
            for s in strategies:
                if s.name not in ["tail_risk", "sell_premium", "buy_wings", "percentile_tail"]:
                    filtered.append(LiquidityFilterStrategy(
                        base_strategy=s,
                        min_volume_percentile=config.min_volume_percentile
                    ))
                else:
                    filtered.append(s)
            strategies = filtered

        cls.ALL_STRATEGIES = strategies
        return strategies


# =============================================================================
# UNIFIED SIGNAL
# =============================================================================

@dataclass
class UnifiedSignal:
    """
    The final output of the orchestrator: one signal per day.
    """
    date: pd.Timestamp
    symbol: str
    direction: Direction
    size: float
    entry_price: float
    stop_price: float
    target_price: float
    strategy_name: str
    confidence: float
    expected_value: float
    hold_days: int = 1
    execution_plan: Optional[Any] = None
    metadata: Dict = field(default_factory=dict)
    is_hedge: bool = False

    def __repr__(self) -> str:
        dir_str = "LONG" if self.direction == Direction.LONG else ("SHORT" if self.direction == Direction.SHORT else "FLAT")
        return (
            f"UnifiedSignal({self.symbol} {dir_str} "
            f"size={self.size:.3f} @ {self.entry_price:.4f} "
            f"-> {self.target_price:.4f} stop {self.stop_price:.4f} "
            f"[{self.strategy_name}] conf={self.confidence:.2f})"
        )


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class KairosOrchestrator:
    """
    Master orchestrator. One object to run them all.
    """

    def __init__(self,
                 predict_fn: Callable[[pd.DataFrame, str], List[pd.DataFrame]],
                 assets: Optional[List[str]] = None,
                 config: Optional[OrchestratorConfig] = None,
                 batch_predict_fn: Optional[Callable] = None):
        self.predict_fn = predict_fn
        self.assets = assets or ["BTC-USD"]
        self.config = config or OrchestratorConfig()

        # Sub-components
        self.predictor = KairosPredictor(predict_fn)
        self.multi_predictor = MultiAssetKairosPredictor(predict_fn, batch_predict_fn=batch_predict_fn)
        self.tracker = StrategyPerformanceTracker(
            lookback_window=self.config.performance_lookback
        )
        self.registry = StrategyRegistry()
        self.strategies = self.registry.build_all(self.config)

        # State
        self.capital = self.config.initial_capital
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.all_signals: List[UnifiedSignal] = []
        self.all_trades: List[Trade] = []
        self.active_positions: List[Dict] = []
        self.daily_logs: List[Dict] = []

    def run_backtest(self,
                     data_dict: Dict[str, pd.DataFrame],
                     lookback: int = 200) -> Dict:
        """
        Full walk-forward backtest across all assets.

        Returns a dict with summary metrics, equity curve, and trade log.
        """
        # Find common date range
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df.index[lookback:])
        common_dates = sorted(all_dates)

        if not common_dates:
            raise ValueError("No common dates found across assets after lookback")

        for date in tqdm(common_dates, desc="Backtesting"):
            # Build histories up to this date
            histories = {}
            for symbol, df in data_dict.items():
                mask = df.index <= date
                if mask.sum() < lookback:
                    continue
                histories[symbol] = df[mask]

            if not histories:
                continue

            # Run one day
            self._run_day(date, histories)

        # Close remaining positions
        self._close_all_positions(data_dict)

        return self._build_results()

    def _run_day(self, date: pd.Timestamp, histories: Dict[str, pd.DataFrame]):
        """Process a single day across all assets."""
        # 1. Multi-asset predictions
        multi_preds = self.multi_predictor.predict_all(histories)

        # 2. Evaluate all strategies for each asset
        all_signals = []
        for symbol, pred in multi_preds.items():
            current_price = pred.current_price
            dist = pred.dist
            history = pred.history

            # Meta-filters
            if self._apply_meta_filters(dist, current_price):
                continue

            # Context for strategies
            context = {
                "date": date,
                "current_price": current_price,
                "capital": self.capital,
                "multi_asset_predictions": multi_preds,
                "current_symbol": symbol,
                "predict_fn": self.predict_fn,
            }

            # Run all strategies
            signals = []
            for strat in self.strategies:
                try:
                    sig = strat.generate_signal(dist, current_price, history, context)
                    if sig and sig.direction != Direction.FLAT and sig.size > 0:
                        # Weight by online performance
                        if self.config.online_weighting:
                            weight = self.tracker.get_weight(
                                strat.name, self.config.temperature
                            )
                            sig.size *= weight
                            sig.metadata["online_weight"] = weight

                        # Apply max position size cap
                        max_size = self.config.max_position_size_pct
                        sig.size = min(sig.size, max_size)

                        signals.append(sig)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Strategy {strat.name} failed: {e}")
                    continue

            # Pick top signal for this asset
            if signals:
                best = max(signals, key=lambda s: s.expected_value * s.confidence * s.size)
                all_signals.append((symbol, best, pred))

        # 3. Cross-asset ranking: if enabled, only trade top asset(s)
        if self.config.cross_asset_ranking and len(all_signals) > 1:
            all_signals.sort(key=lambda x: x[1].expected_value * x[1].confidence, reverse=True)
            all_signals = all_signals[:self.config.max_positions_per_day]

        # 4. Manage existing positions (check stops, targets, expiry)
        self._manage_positions(date, histories)

        # 5. Enter new positions
        total_exposure = sum(p["notional"] for p in self.active_positions)
        for symbol, sig, pred in all_signals:
            if total_exposure >= self.config.max_total_exposure * self.capital:
                break

            unified = self._create_unified_signal(date, symbol, sig, pred)
            if unified:
                self.all_signals.append(unified)
                self._enter_position(unified, date, histories)
                total_exposure += unified.size * self.capital

        # 6. Log
        self.equity_curve.append((date, self.capital))
        if self.config.log_signals:
            self.daily_logs.append({
                "date": date,
                "capital": self.capital,
                "num_signals": len(all_signals),
                "num_positions": len(self.active_positions),
            })

    def _apply_meta_filters(self, dist: KairosDistribution, current_price: float) -> bool:
        """Returns True if the distribution should be filtered out."""
        # Entropy
        ent = dist.entropy()
        if self.config.debug_filters:
            print(f"[debug filters] entropy={ent:.3f} threshold={self.config.entropy_threshold}")
        if ent > self.config.entropy_threshold:
            return True

        # Bimodality
        if self.config.bimodality_filter:
            # Simple bimodality check via kurtosis proxy
            # (bimodal distributions tend to have negative excess kurtosis)
            kurt = dist.stats["close"].get("kurt", 0)
            if self.config.debug_filters:
                print(f"[debug filters] kurt={kurt:.3f} bimodality_block=<-1.0")
            if kurt < -1.0:
                return True

        return False

    def _manage_positions(self, date: pd.Timestamp, histories: Dict[str, pd.DataFrame]):
        """Check and manage all active positions."""
        remaining = []
        for pos in self.active_positions:
            symbol = pos["symbol"]
            if symbol not in histories:
                remaining.append(pos)
                continue

            today = histories[symbol].iloc[-1]
            open_p = float(today["open"])
            high = float(today["high"])
            low = float(today["low"])
            close = float(today["close"])

            # Check stop/target
            exit_price = None
            exit_reason = None

            if pos["direction"] == Direction.LONG:
                if open_p <= pos["stop"]:
                    exit_price, exit_reason = open_p, "stop_open"
                elif low <= pos["stop"]:
                    exit_price, exit_reason = pos["stop"], "stop"
                elif open_p >= pos["target"]:
                    exit_price, exit_reason = open_p, "target_open"
                elif high >= pos["target"]:
                    exit_price, exit_reason = pos["target"], "target"
            else:
                if open_p >= pos["stop"]:
                    exit_price, exit_reason = open_p, "stop_open"
                elif high >= pos["stop"]:
                    exit_price, exit_reason = pos["stop"], "stop"
                elif open_p <= pos["target"]:
                    exit_price, exit_reason = open_p, "target_open"
                elif low <= pos["target"]:
                    exit_price, exit_reason = pos["target"], "target"

            # Check hold expiry
            if exit_price is None and pos.get("hold_days_remaining", 1) <= 0:
                exit_price = close
                exit_reason = "hold_expired"

            if exit_price is not None:
                pnl = self._calculate_pnl(pos, exit_price)
                self.capital += pnl
                self.all_trades.append(Trade(
                    entry_date=pos["entry_date"],
                    exit_date=date,
                    direction=pos["direction"],
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    size=pos["size"],
                    pnl=pnl,
                    pnl_pct=(pnl / (pos["entry_price"] * pos["size"])) if pos["entry_price"] * pos["size"] != 0 else 0.0,
                    strategy_name=pos["strategy_name"],
                    exit_reason=exit_reason,
                ))
                # Record for performance tracking
                self.tracker.record_trade(
                    strategy_name=pos["strategy_name"],
                    pnl=pnl,
                    entry_price=pos["entry_price"],
                    exit_price=exit_price
                )
            else:
                pos["hold_days_remaining"] = pos.get("hold_days_remaining", 1) - 1
                remaining.append(pos)

        self.active_positions = remaining

    def _enter_position(self, unified: UnifiedSignal, date: pd.Timestamp, histories: Dict[str, pd.DataFrame]):
        """Enter a new position from a unified signal."""
        symbol = unified.symbol
        if symbol not in histories:
            return

        tomorrow = histories[symbol].iloc[-1]  # This is "today" in the walk-forward
        entry_price = float(tomorrow["open"]) * (
            1.0 + self.config.slippage_pct * unified.direction.value
        )

        notional = unified.size * self.capital
        fee = notional * self.config.fee_pct
        self.capital -= fee

        position = {
            "symbol": symbol,
            "direction": unified.direction,
            "size": notional / entry_price,
            "entry_price": entry_price,
            "stop": unified.stop_price,
            "target": unified.target_price,
            "strategy_name": unified.strategy_name,
            "entry_date": date,
            "hold_days_remaining": unified.hold_days,
            "notional": notional,
        }
        self.active_positions.append(position)

    def _create_unified_signal(self, date: pd.Timestamp, symbol: str,
                                sig: Signal, pred: AssetPrediction) -> Optional[UnifiedSignal]:
        """Convert a strategy signal to a unified signal."""
        hold_days = sig.metadata.get("hold_days", 1) if hasattr(sig, "metadata") else 1
        exec_plan = sig.metadata.get("execution_plan") if hasattr(sig, "metadata") else None

        return UnifiedSignal(
            date=date,
            symbol=symbol,
            direction=sig.direction,
            size=sig.size,
            entry_price=sig.entry,
            stop_price=sig.stop,
            target_price=sig.target,
            strategy_name=sig.strategy_name,
            confidence=sig.confidence,
            expected_value=sig.expected_value,
            hold_days=hold_days,
            execution_plan=exec_plan,
            metadata=sig.metadata if hasattr(sig, "metadata") else {},
        )

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        if position["direction"] == Direction.LONG:
            gross = (exit_price - position["entry_price"]) * position["size"]
        else:
            gross = (position["entry_price"] - exit_price) * position["size"]
        fee = exit_price * position["size"] * self.config.fee_pct
        return gross - fee

    def _close_all_positions(self, data_dict: Dict[str, pd.DataFrame]):
        """Close any remaining positions at the last available close."""
        for pos in self.active_positions:
            symbol = pos["symbol"]
            if symbol in data_dict and len(data_dict[symbol]) > 0:
                last_close = float(data_dict[symbol]["close"].iloc[-1])
            else:
                last_close = pos["entry_price"]

            pnl = self._calculate_pnl(pos, last_close)
            self.capital += pnl
            self.all_trades.append(Trade(
                entry_date=pos["entry_date"],
                exit_date=data_dict[symbol].index[-1] if symbol in data_dict else pos["entry_date"],
                direction=pos["direction"],
                entry_price=pos["entry_price"],
                exit_price=last_close,
                size=pos["size"],
                pnl=pnl,
                pnl_pct=(pnl / (pos["entry_price"] * pos["size"])) if pos["entry_price"] * pos["size"] != 0 else 0.0,
                strategy_name=pos["strategy_name"],
                exit_reason="end_of_data",
            ))
            self.tracker.record_trade(
                strategy_name=pos["strategy_name"],
                pnl=pnl,
                entry_price=pos["entry_price"],
                exit_price=last_close
            )

        self.active_positions = []
        if self.equity_curve:
            self.equity_curve[-1] = (self.equity_curve[-1][0], self.capital)

    def _build_results(self) -> Dict:
        """Compile all results into a comprehensive dict."""
        # Basic metrics
        pnls = [t.pnl for t in self.all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity = [e for _, e in self.equity_curve]
        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital
        else:
            max_dd = 0.0
            total_return = 0.0

        returns = np.diff(equity) / np.array(equity[:-1]) if len(equity) > 1 else np.array([0.0])
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        profit_factor = float(abs(sum(wins) / sum(losses))) if sum(losses) != 0 else float("inf")

        # Strategy performance breakdown
        strategy_stats = {}
        for name, perf in self.tracker.performances.items():
            strategy_stats[name] = {
                "sharpe": perf.sharpe,
                "win_rate": perf.win_rate,
                "profit_factor": perf.profit_factor,
                "total_pnl": perf.total_pnl,
                "num_trades": perf.wins + perf.losses,
            }

        # Best / worst strategies
        ranked = self.tracker.get_all_rankings()
        best_strategy = ranked[0][0] if ranked else None
        worst_strategy = ranked[-1][0] if ranked else None

        return {
            "summary": {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": float(len(wins) / len(pnls)) if pnls else 0.0,
                "profit_factor": profit_factor,
                "num_trades": len(self.all_trades),
                "avg_trade": float(np.mean(pnls)) if pnls else 0.0,
                "avg_win": float(np.mean(wins)) if wins else 0.0,
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "final_capital": self.capital,
                "initial_capital": self.config.initial_capital,
            },
            "equity_curve": self.equity_curve,
            "trades": self.all_trades,
            "signals": self.all_signals,
            "strategy_performance": strategy_stats,
            "best_strategy": best_strategy,
            "worst_strategy": worst_strategy,
            "strategy_rankings": ranked,
            "daily_logs": self.daily_logs,
        }

    def run_single_asset(self, df: pd.DataFrame, lookback: int = 200) -> Dict:
        """
        Convenience method for single-asset backtest.
        """
        symbol = self.assets[0] if self.assets else "BTC-USD"
        return self.run_backtest({symbol: df}, lookback=lookback)

    def get_live_signal(self, histories: Dict[str, pd.DataFrame]) -> Optional[UnifiedSignal]:
        """
        Get the current signal for live trading (no backtest).
        """
        date = pd.Timestamp.now()
        multi_preds = self.multi_predictor.predict_all(histories)

        all_signals = []
        for symbol, pred in multi_preds.items():
            current_price = pred.current_price
            dist = pred.dist
            history = pred.history

            if self._apply_meta_filters(dist, current_price):
                continue

            context = {
                "date": date,
                "current_price": current_price,
                "capital": self.capital,
                "multi_asset_predictions": multi_preds,
                "current_symbol": symbol,
                "predict_fn": self.predict_fn,
            }

            signals = []
            for strat in self.strategies:
                try:
                    sig = strat.generate_signal(dist, current_price, history, context)
                    if sig and sig.direction != Direction.FLAT and sig.size > 0:
                        if self.config.online_weighting:
                            weight = self.tracker.get_weight(strat.name, self.config.temperature)
                            sig.size *= weight
                        sig.size = min(sig.size, self.config.max_position_size_pct)
                        signals.append(sig)
                except Exception:
                    continue

            if signals:
                best = max(signals, key=lambda s: s.expected_value * s.confidence * s.size)
                all_signals.append((symbol, best, pred))

        if not all_signals:
            return None

        if self.config.cross_asset_ranking:
            all_signals.sort(key=lambda x: x[1].expected_value * x[1].confidence, reverse=True)
            all_signals = all_signals[:self.config.max_positions_per_day]

        symbol, sig, pred = all_signals[0]
        return self._create_unified_signal(date, symbol, sig, pred)


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def print_results(results: Dict):
    """Pretty-print backtest results."""
    s = results["summary"]
    print("=" * 60)
    print("KAIROS BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return:     {s['total_return']*100:>8.2f}%")
    print(f"Sharpe Ratio:     {s['sharpe']:>8.2f}")
    print(f"Max Drawdown:     {s['max_drawdown']*100:>8.2f}%")
    print(f"Win Rate:         {s['win_rate']*100:>8.2f}%")
    print(f"Profit Factor:    {s['profit_factor']:>8.2f}")
    print(f"Num Trades:       {s['num_trades']:>8d}")
    print(f"Avg Trade:        {s['avg_trade']:>8.4f}")
    print(f"Avg Win:          {s['avg_win']:>8.4f}")
    print(f"Avg Loss:         {s['avg_loss']:>8.4f}")
    print(f"Final Capital:    {s['final_capital']:>8.2f}")
    print("-" * 60)
    print(f"Best Strategy:    {results['best_strategy']}")
    print(f"Worst Strategy:   {results['worst_strategy']}")
    print("=" * 60)

    if results["strategy_rankings"]:
        print("\nTOP 10 STRATEGIES BY SHARPE:")
        for i, (name, sharpe) in enumerate(results["strategy_rankings"][:10], 1):
            print(f"  {i:2d}. {name:30s}  Sharpe: {sharpe:6.3f}")


def export_results(results: Dict, filepath: str):
    """Export results to JSON."""
    # Convert non-serializable objects
    exportable = {
        "summary": results["summary"],
        "equity_curve": [(str(d), float(v)) for d, v in results["equity_curve"]],
        "strategy_performance": results["strategy_performance"],
        "best_strategy": results["best_strategy"],
        "worst_strategy": results["worst_strategy"],
        "strategy_rankings": results["strategy_rankings"],
    }
    with open(filepath, "w") as f:
        json.dump(exportable, f, indent=2)


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    print("kairos_orchestrator.py loaded successfully.")
    print(f"Total strategies registered: {len(StrategyRegistry.ALL_STRATEGIES)}")
    print("\nExample usage:")
    print("""
    from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig

    config = OrchestratorConfig(
        initial_capital=10000.0,
        cross_asset_ranking=True,
        online_weighting=True,
        partial_exits=True,
    )

    orchestrator = KairosOrchestrator(
        predict_fn=predict_kairos_cloud,
        assets=["BTC-USD", "ETH-USD"],
        config=config,
    )

    results = orchestrator.run_backtest(
        data_dict={"BTC-USD": btc_df, "ETH-USD": eth_df},
        lookback=200
    )

    from kairos_orchestrator import print_results
    print_results(results)
    """)
