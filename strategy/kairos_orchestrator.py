"""
kairos_orchestrator.py
======================
The master orchestrator for the Kairos trading framework.

Wires together strategies across many modules, including the original 42
(kairos_backtest/path/horizon/execution/meta), crypto/forex/stocks/universal
asset-class modules, and 27 awesome-quant strategies added across
kairos_volatility.py, kairos_econometric.py, kairos_ml.py, kairos_sentiment.py,
kairos_execution.py, kairos_meta.py, and kairos_backtest.py.

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
        DecisionTreeRouter, KairosSettings,
        PercentileEntryStrategy, DynamicBracketStrategy, SkewStrategy,
        RangeTradingStrategy, TrendFollowingStrategy, VolatilityArbStrategy,
        HighLowStrategy, OpenGapStrategy, FadeExtremeStrategy,
        MomentumContinuationStrategy, ExpectedValueStrategy,
        MartingaleFloorStrategy, RSIFilterStrategy, MACDFilterStrategy,
        BollingerValidationStrategy, SupportConfluenceStrategy,
        InverseVarianceSizingStrategy, CloseDirectionStrategy,
        VaRPositionCapStrategy, DistributionOverlapStrategy,
        ModelDecayMonitorStrategy, OvernightExposureFilter,
        RSIDivergenceStrategy, LeverageCalibrationStrategy,
        StochasticFilterStrategy, ADXGateStrategy, OBVConfirmationStrategy,
        MTFConsensusStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_backtest: {e}")

try:
    from kairos_path import (
        KairosPathExtractor, PathPattern,
        PathRallyStrategy, PathFadeStrategy,
        PathVShapeStrategy, PathInvertedVStrategy,
        PathHighLowSequenceStrategy,
        ConditionalPathProbabilityStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_path: {e}")

try:
    from kairos_horizon import (
        KairosMultiHorizonPredictor, HorizonStack,
        MultiHorizonHoldStrategy, ConfidenceDecayFilterStrategy,
        RollingHorizonStrategy, MultiHorizonBacktestEngine,
        PathIntegrationStrategy,
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
        PyramidingStrategy, TimeBasedStopStrategy,
        VolumeProfileLevelsStrategy, CVDDivergenceStrategy,
        TWAPExecutionStrategy, ImplementationShortfallStrategy,
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
        RegimeClusterStrategy, MonteCarloScenarioStrategy,
        MultiFactorRankStrategy, PCAResidualReversalStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_meta: {e}")

try:
    from kairos_crypto import (
        FundingRateArbitrage, BasisTrade, StablecoinDepeg,
        ExchangeSpreadArbitrage, LiquidationFrontRun, FundingRatePrediction,
        OnChainFlowFilter, GammaSqueeze, HashRateFilter, FundingHarvest,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_crypto: {e}")

try:
    from kairos_forex import (
        CarryTrade, SessionBreakout, LondonFixFade, CBDivergence,
        SafeHavenRotation, TriangularArbitrage, CDSSpreadFilter,
        COTPositioningFilter, AsianRangeBreakout, OISSwapSpread,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_forex: {e}")

try:
    from kairos_stocks import (
        PEAD, EarningsMomentum, IndexRebalance, SectorRotation,
        CointegrationPairs, MergerArb, BuybackYield, ShortSqueeze,
        InsiderCluster, DarkPoolFilter, BuybackDrift, DividendCapture,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_stocks: {e}")

try:
    from kairos_volatility import (
        ATRBracketStrategy, GARCHFilterStrategy, VolTargetSizerStrategy,
        VarianceRiskPremiumStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_volatility: {e}")

try:
    from kairos_econometric import (
        ARIMADisagreementStrategy, VARLeadLagStrategy, SeasonalityFilterStrategy,
        ChangepointGuardStrategy, GrangerPairsStrategy, MatrixProfileAnomalyStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_econometric: {e}")

try:
    from kairos_ml import (
        MetaLabelStrategy, GBMDirectionStrategy, LPPLSGuardStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_ml: {e}")

try:
    from kairos_sentiment import (
        NewsSentimentFilterStrategy, Institutional13FFilterStrategy,
        SocialMomentumStrategy, EconCalendarGuardStrategy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_sentiment: {e}")

try:
    from kairos_universal import (
        KalmanPairs, HurstRegimeSwitch, CopulaPairs, CointegrationECT,
        HMMRegime, WaveletMomentum, DFAPersistence, TransferEntropy,
        GNNSectorRotation, RLMetaController, FractalDimension, LZComplexity,
        RQADeterminism, MutualInformationWeight, GaussianProcess,
        BSTSDecomposition, ParticleFilter, SpectralClustering,
    )
except ImportError as e:
    raise ImportError(f"Failed to import kairos_universal: {e}")


# =============================================================================
# SHARPE HELPERS
# =============================================================================

# Minimum sample size before a Sharpe ratio is considered statistically
# meaningful. Below this, near-zero variance can blow up the ratio
# (e.g. n=2 with two nearly-identical pnl values -> Sharpe of 1e15+).
MIN_SIGNALS_FOR_SHARPE = 3
# Floor applied to the std-dev denominator to avoid division by ~0.
_SHARPE_STD_EPSILON = 1e-9
# Hard clamp so a single pathological case can never dominate rankings.
_SHARPE_CLAMP = 100.0


def _safe_sharpe(returns: np.ndarray, annualization_factor: float, min_n: int = MIN_SIGNALS_FOR_SHARPE) -> float:
    """Compute an annualised Sharpe ratio that is robust to tiny/degenerate samples.

    Returns 0.0 if there are fewer than `min_n` observations (insufficient
    sample to estimate variance reliably). Otherwise floors the std-dev
    denominator with a small epsilon and clamps the result to
    [-_SHARPE_CLAMP, _SHARPE_CLAMP].
    """
    n = len(returns)
    if n < min_n:
        return 0.0
    std_r = float(np.std(returns))
    if std_r <= 0:
        return 0.0
    std_r = max(std_r, _SHARPE_STD_EPSILON)
    sharpe = float(np.mean(returns) / std_r * annualization_factor)
    return float(np.clip(sharpe, -_SHARPE_CLAMP, _SHARPE_CLAMP))


@dataclass
class OrchestratorConfig:
    """Configuration for the Kairos orchestrator."""
    _kwargs = None

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

    # Baseline mode
    no_prediction: bool = False

    # Disabled strategies (shadow-tested and found unprofitable even with perfect predictions)
    disabled_strategies: Set[str] = field(default_factory=lambda: {
        "skew", "multi_horizon_hold", "dynamic_bracket", "inverse_variance",
        "rolling_horizon", "percentile_entry", "path_v_shape", "path_execution",
        "distribution_overlap", "volume_fade", "tail_asymmetry", "rsi_filter",
        "path_high_low_sequence", "range_trading", "volume_confirmation",
        "cross_asset_momentum_transfer", "momentum_continuation",
        "trend_following", "open_gap",
    })


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

class StrategyRegistry:
    """
    Maintains the full registry of all strategies (108 after disabled-strategy
    filtering; 121 constructed before filtering).
    """

    ALL_STRATEGIES: List[Strategy] = []

    def __init__(self):
        # Instance-level: at most one active allocator per registry.
        self._allocator: Optional["PortfolioAllocator"] = None

    def register_allocator(self, allocator: Optional["PortfolioAllocator"]) -> None:
        """
        Register the active portfolio allocator.

        At most one allocator is active at a time; calling this again replaces
        the previously registered allocator. Passing None clears it, disabling
        allocator-driven sizing (per-signal fallback resumes).
        """
        self._allocator = allocator

    def get_allocator(self) -> Optional["PortfolioAllocator"]:
        """Return the currently registered allocator, or None if none is set."""
        return self._allocator

    @classmethod
    def build_all(cls, config: OrchestratorConfig) -> List[Strategy]:
        """Build all 108 strategies with the given config."""
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

        # === NEW STRATEGIES (12) ===
        # Standalone strategies
        strategies.extend([
            DistributionOverlapStrategy(),
            ConditionalPathProbabilityStrategy(),
            RSIDivergenceStrategy(),
            PathIntegrationStrategy(max_horizon=config.max_horizon),
        ])
        # Wrappers - use TrendFollowingStrategy so close_direction can be disabled
        base_trend = (CloseDirectionStrategy()
                      if "close_direction" not in config.disabled_strategies
                      else TrendFollowingStrategy())
        strategies.extend([
            VaRPositionCapStrategy(base_strategy=base_trend),
            LeverageCalibrationStrategy(base_strategy=base_trend),
            OvernightExposureFilter(base_strategy=base_trend),
            ModelDecayMonitorStrategy(base_strategy=base_trend),
            PyramidingStrategy(base_strategy=base_trend),
            TimeBasedStopStrategy(base_strategy=base_trend),
        ])
        # Meta-selectors across all base strategies (exclude disabled ones)
        base_pool = [s for s in [TrendFollowingStrategy(), RangeTradingStrategy(),
                                  SkewStrategy(), MomentumContinuationStrategy()]
                     if s.name not in config.disabled_strategies]
        strategies.extend([
            RegimeClusterStrategy(base_strategies=base_pool),
            MonteCarloScenarioStrategy(base_strategies=base_pool),
        ])

        # === CRYPTO STRATEGIES (10) ===
        strategies.extend([
            FundingRateArbitrage(),
            BasisTrade(),
            StablecoinDepeg(),
            ExchangeSpreadArbitrage(),
            LiquidationFrontRun(),
            FundingRatePrediction(),
            OnChainFlowFilter(base_strategy=TrendFollowingStrategy()),
            GammaSqueeze(),
            HashRateFilter(base_strategy=TrendFollowingStrategy()),
            FundingHarvest(),
        ])

        # === FOREX STRATEGIES (10) ===
        strategies.extend([
            CarryTrade(),
            SessionBreakout(),
            LondonFixFade(),
            CBDivergence(),
            SafeHavenRotation(),
            TriangularArbitrage(),
            CDSSpreadFilter(base_strategy=TrendFollowingStrategy()),
            COTPositioningFilter(base_strategy=TrendFollowingStrategy()),
            AsianRangeBreakout(),
            OISSwapSpread(),
        ])

        # === STOCK STRATEGIES (12) ===
        strategies.extend([
            PEAD(),
            EarningsMomentum(),
            IndexRebalance(),
            SectorRotation(),
            CointegrationPairs(),
            MergerArb(),
            BuybackYield(),
            ShortSqueeze(base_strategy=MomentumContinuationStrategy()),
            InsiderCluster(base_strategy=TrendFollowingStrategy()),
            DarkPoolFilter(base_strategy=TrendFollowingStrategy()),
            BuybackDrift(),
            DividendCapture(),
        ])

        # === UNIVERSAL STRATEGIES (18) ===
        strategies.extend([
            KalmanPairs(),
            HurstRegimeSwitch(),
            CopulaPairs(),
            CointegrationECT(),
            HMMRegime(),
            WaveletMomentum(),
            DFAPersistence(),
            TransferEntropy(),
            GNNSectorRotation(),
            RLMetaController(all_strategies=[
                TrendFollowingStrategy(), MomentumContinuationStrategy(),
                RangeTradingStrategy(), SkewStrategy(),
            ]),
            FractalDimension(base_strategy=TrendFollowingStrategy()),
            LZComplexity(base_strategy=TrendFollowingStrategy()),
            RQADeterminism(),
            MutualInformationWeight(feature_map={
                "rsi_filter": "rsi",
                "volume_confirmation": "volume",
                "trend_following": "momentum",
                "momentum_continuation": "momentum",
            }),
            GaussianProcess(base_strategy=TrendFollowingStrategy()),
            BSTSDecomposition(),
            ParticleFilter(base_strategy=TrendFollowingStrategy()),
            SpectralClustering(),
        ])

        # === AWESOME-QUANT STRATEGIES (27) ===
        # Standalone strategies - default constructor params
        strategies.extend([
            VarianceRiskPremiumStrategy(),
            VARLeadLagStrategy(),
            GrangerPairsStrategy(),
            MatrixProfileAnomalyStrategy(),
            GBMDirectionStrategy(),
            SocialMomentumStrategy(),
            CVDDivergenceStrategy(),
            MultiFactorRankStrategy(),
            PCAResidualReversalStrategy(),
        ])

        # Wrappers - paired with a base strategy that complements their intent.
        stochastic_filter = StochasticFilterStrategy(base_strategy=FadeExtremeStrategy())

        adx_gate_trend = ADXGateStrategy(base_strategy=TrendFollowingStrategy(), kind="trend")
        adx_gate_trend.name = "adx_gate_trend"
        adx_gate_reversion = ADXGateStrategy(base_strategy=RangeTradingStrategy(), kind="reversion")
        adx_gate_reversion.name = "adx_gate_reversion"

        obv_confirmation = OBVConfirmationStrategy(base_strategy=MomentumContinuationStrategy())
        mtf_consensus = MTFConsensusStrategy(base_strategy=TrendFollowingStrategy())
        atr_bracket = ATRBracketStrategy(base_strategy=TrendFollowingStrategy())
        garch_filter = GARCHFilterStrategy(base_strategy=TrendFollowingStrategy())
        vol_target_sizer = VolTargetSizerStrategy(base_strategy=ExpectedValueStrategy())
        arima_disagreement = ARIMADisagreementStrategy(base_strategy=TrendFollowingStrategy())
        seasonality_filter = SeasonalityFilterStrategy(base_strategy=TrendFollowingStrategy())
        changepoint_guard = ChangepointGuardStrategy(base_strategy=TrendFollowingStrategy())
        meta_label = MetaLabelStrategy(base_strategy=ExpectedValueStrategy())
        lppls_guard = LPPLSGuardStrategy(base_strategy=TrendFollowingStrategy())
        news_sentiment_filter = NewsSentimentFilterStrategy(base_strategy=MomentumContinuationStrategy())
        institutional_13f_filter = Institutional13FFilterStrategy(base_strategy=TrendFollowingStrategy())
        econ_calendar_guard = EconCalendarGuardStrategy(base_strategy=TrendFollowingStrategy())
        volume_profile_levels = VolumeProfileLevelsStrategy(base_strategy=TrendFollowingStrategy())
        twap_execution = TWAPExecutionStrategy(base_strategy=MomentumContinuationStrategy())
        implementation_shortfall = ImplementationShortfallStrategy(base_strategy=TrendFollowingStrategy())

        strategies.extend([
            stochastic_filter,
            adx_gate_trend,
            adx_gate_reversion,
            obv_confirmation,
            mtf_consensus,
            atr_bracket,
            garch_filter,
            vol_target_sizer,
            arima_disagreement,
            seasonality_filter,
            changepoint_guard,
            meta_label,
            lppls_guard,
            news_sentiment_filter,
            institutional_13f_filter,
            econ_calendar_guard,
            volume_profile_levels,
            twap_execution,
            implementation_shortfall,
        ])

        # Drop strategies the user has disabled
        if config.disabled_strategies:
            strategies = [s for s in strategies if s.name not in config.disabled_strategies]

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


def apply_allocator(
    signals: Dict[str, "Signal"],
    allocator: Optional["PortfolioAllocator"],
    returns: Optional[pd.DataFrame],
    dists: Dict[str, Any],
    context: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, "Signal"]:
    """
    Apply a registered PortfolioAllocator to a set of surviving per-symbol
    signals, replacing each signal's size with min(original_size, |weight|).

    Per §7.3 of the design doc: allocation runs *after* per-asset signal
    generation and meta-filters. Per-signal size is preserved as the
    within-asset cap - the allocator can only shrink it, never widen it.
    Symbols whose allocator weight is exactly 0 are dropped entirely.

    Graceful degradation: if no allocator is registered, fewer than 2 signals
    survived (nothing to allocate across), or the allocator raises, the
    original signals are returned unchanged so the orchestrator keeps trading.

    Args:
        signals: Dict[symbol -> Signal], the surviving best-per-asset signals.
        allocator: The active PortfolioAllocator, or None.
        returns: Trailing daily returns panel (context["returns_window"]).
        dists: Dict[symbol -> KairosDistribution].
        context: Execution context dict passed through to allocate().
        verbose: If True, print a warning on allocator failure.

    Returns:
        Dict[symbol -> Signal] with sizes replaced (and zero-weight symbols
        dropped), or the original `signals` dict unchanged on any failure.
    """
    if allocator is None or len(signals) <= 1:
        return signals

    try:
        weights = allocator.allocate(signals, returns, dists, context)
    except Exception as e:
        if verbose:
            print(f"Allocator {getattr(allocator, 'name', allocator)} failed: {e}")
        return signals

    result: Dict[str, "Signal"] = {}
    for symbol, sig in signals.items():
        if symbol not in weights:
            # Allocator didn't opine on this symbol - keep it as-is.
            result[symbol] = sig
            continue
        w = weights[symbol]
        if w == 0:
            continue
        sig.size = min(sig.size, abs(w))
        result[symbol] = sig
    return result


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
                 batch_predict_fn: Optional[Callable] = None, **kwargs):

        self._kwargs = kwargs
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
        self._prev_dist: Dict[str, KairosDistribution] = {}
        # Shadow tracking: (date, symbol, strategy_name, direction, stop, target)
        self._shadow_signals: List[Tuple] = []
        self._shadow_seen: set = set()

    def run_backtest(self,
                     data_dict: Dict[str, pd.DataFrame],
                     lookback: int = 200) -> Dict:
        """
        Full walk-forward backtest across all assets.

        Returns a dict with summary metrics, equity curve, and trade log.
        """
        self._data_dict = data_dict
        self._lookback = lookback
        self._shadow_signals = []
        self._shadow_seen = set()

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

    def _make_realized_predictions(self, date: pd.Timestamp,
                                    histories: Dict[str, pd.DataFrame]) -> Dict:
        """Oracle baseline: build AssetPrediction from the actual next bar."""
        result = {}
        for symbol, history in histories.items():
            full_df = self._data_dict.get(symbol)
            future = full_df[full_df.index > date] if full_df is not None else pd.DataFrame()
            bar = future.iloc[0] if not future.empty else history.iloc[-1]
            current_price = float(history.iloc[-1]["close"])
            dist = KairosDistribution.from_bar(
                bar, n_samples=KairosSettings.pred_samples, center=current_price
            )
            result[symbol] = AssetPrediction(
                symbol=symbol,
                dist=dist,
                current_price=current_price,
                history=history,
            )
        return result

    def _compute_returns_window(self, histories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build the trailing daily log-return panel for the active universe.

        Computed once per day from the per-asset histories the orchestrator
        already holds (no extra data fetch). Columns are asset symbols, index
        is dates common to all assets (inner join), values are close-to-close
        log returns. Used by allocators (context["returns_window"]) and by
        any §2/§4/§6.4 portfolio/econometric/factor strategy.
        """
        if not histories:
            return pd.DataFrame()

        series = {}
        for symbol, history in histories.items():
            if history is None or len(history) < 2 or "close" not in history.columns:
                continue
            close = history["close"].astype(float)
            log_ret = np.log(close / close.shift(1))
            series[symbol] = log_ret

        if not series:
            return pd.DataFrame()

        panel = pd.DataFrame(series).dropna(how="all")
        return panel

    def _compute_realized_vol(self, returns_window: pd.DataFrame,
                               window: int = 20) -> Dict[str, float]:
        """
        Per-symbol trailing realized volatility (std of daily log returns
        over the last `window` observations), computed once per day from
        `returns_window`.
        """
        if returns_window is None or returns_window.empty:
            return {}

        realized_vol = {}
        for symbol in returns_window.columns:
            col = returns_window[symbol].dropna().tail(window)
            if len(col) < 2:
                continue
            realized_vol[symbol] = float(col.std())
        return realized_vol

    def _run_day(self, date: pd.Timestamp, histories: Dict[str, pd.DataFrame]):
        """Process a single day across all assets."""
        # 1. Multi-asset predictions
        if self.config.no_prediction:
            multi_preds = self._make_realized_predictions(date, histories)
        else:
            multi_preds = self.multi_predictor.predict_all(histories)

        # 1b. Context enrichment computed once per day for the active universe:
        # trailing daily returns panel and per-symbol realized vol. Cheap
        # (pandas pct_change/std only) - safe to compute every day.
        returns_window = self._compute_returns_window(histories)
        realized_vol = self._compute_realized_vol(returns_window)

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
                "predict_fn": (lambda *a, **kw: []) if self.config.no_prediction else self.predict_fn,
                "prev_dist": self._prev_dist.get(symbol),
                "current_position": next(
                    (p for p in self.active_positions if p["symbol"] == symbol), None
                ),
                "bar_index": len(self.equity_curve),
                "returns_window": returns_window,
                "realized_vol": realized_vol,
            }

            # Run all strategies
            signals = []
            for strat in self.strategies:
                try:
                    sig = strat.generate_signal(dist, current_price, history, context)
                    if sig and sig.direction != Direction.FLAT and sig.size > 0:
                        # Shadow tracking: record signal before competitive weighting.
                        # Use sig.strategy_name (not strat.name) - LiquidityFilter and
                        # other wrappers preserve the inner signal's name, so this
                        # correctly identifies the originating strategy.
                        # Deduplicate per (date, symbol, strategy_name): wrapper chains
                        # like VaRPositionCap→CloseDirection emit strategy_name=
                        # "close_direction" up to 7× per slot; only the first counts.
                        # Also record sig.entry so _compute_shadow_performance can
                        # convert stop/target to % offsets (fixes cross-asset strategies
                        # whose stop/target are in a different asset's price units).
                        _shadow_key = (date, symbol, sig.strategy_name)
                        if _shadow_key not in self._shadow_seen:
                            self._shadow_seen.add(_shadow_key)
                            self._shadow_signals.append((
                                date, symbol,
                                sig.strategy_name, sig.direction,
                                sig.stop, sig.target, sig.entry,
                            ))

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

            # Store distribution for next day's DistributionOverlapStrategy
            self._prev_dist[symbol] = dist

            # Update calibration-tracking strategies
            for strat in self.strategies:
                if hasattr(strat, "update_calibration"):
                    strat.update_calibration(dist, current_price)

            # Pick top signal for this asset
            if signals:
                best = max(signals, key=lambda s: s.expected_value * s.confidence * s.size)
                all_signals.append((symbol, best, pred))

        # 2.5. Portfolio allocator: applied after per-asset signal generation
        # and meta-filters, before cross-asset ranking / position entry.
        # Replaces each surviving signal's size with min(original, |weight|)
        # (per-signal size is the within-asset cap); zero-weight symbols are
        # dropped. Any allocator failure falls back to the original sizes so
        # the orchestrator keeps trading (per-class disabled-strategy
        # fallback semantics from f0662fd).
        allocator = self.registry.get_allocator()
        if allocator is not None and len(all_signals) > 1:
            signals_by_symbol = {symbol: sig for symbol, sig, _ in all_signals}
            dists_by_symbol = {symbol: pred.dist for symbol, _, pred in all_signals}
            allocator_context = {
                "date": date,
                "returns_window": returns_window,
                "realized_vol": realized_vol,
            }
            allocated = apply_allocator(
                signals_by_symbol, allocator, returns_window, dists_by_symbol,
                allocator_context, verbose=self.config.verbose,
            )
            all_signals = [
                (symbol, allocated[symbol], pred)
                for symbol, sig, pred in all_signals
                if symbol in allocated
            ]

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

            # Time-based stop: exit at close if target not reached by time_exit_bar
            if exit_price is None and pos.get("time_exit_bar") is not None:
                bar_index = len(self.equity_curve)
                if bar_index >= pos["time_exit_bar"]:
                    exit_price = close
                    exit_reason = "time_stop"

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
            "time_exit_bar": unified.metadata.get("time_exit_bar"),
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

    def backtest_top_strategies(self, results: Dict, n: int = 3) -> List[Dict]:
        """Build per-strategy stats for the top-n strategies from shadow performance.

        Uses shadow signals (all strategies, evaluated against actual next-bar OHLCV)
        so every strategy that generated signals appears, not just competition winners.
        Metrics are consistent with strategy_rankings because both use shadow data.
        """
        rankings = results.get("strategy_rankings", [])
        shadow_perf = results.get("shadow_performance", {})
        initial_capital = results["summary"]["initial_capital"]

        top_results = []
        for name, _sharpe in rankings[:n]:
            sd = shadow_perf.get(name, {})
            pnl_list: List[float] = sd.get("pnl_list", [])
            sharpe = sd.get("sharpe", 0.0)
            n_signals = sd.get("signal_count", 0)

            # Compound equity curve from pnl_pct (each signal allocated 10% of capital)
            running = initial_capital
            equity_pts: List[float] = [running]
            for r in pnl_list:
                running += running * r * 0.1
                equity_pts.append(running)

            eq = np.array(equity_pts)
            total_ret = (eq[-1] - initial_capital) / initial_capital if len(eq) > 1 else 0.0
            peak = np.maximum.accumulate(eq)
            max_dd = float(((eq - peak) / (peak + 1e-9)).min()) if len(eq) > 1 else 0.0

            wins = [r for r in pnl_list if r > 0]
            losses = [r for r in pnl_list if r <= 0]
            win_rate = len(wins) / n_signals if n_signals else 0.0
            profit_factor = (abs(sum(wins)) / abs(sum(losses))
                             if losses and sum(losses) != 0 else float("inf"))
            total_pnl = sum(pnl_list) * initial_capital * 0.1
            avg_pct_per_trade = float(np.mean(pnl_list)) * 100 if pnl_list else 0.0
            avg_pct_per_win   = float(np.mean(wins))    * 100 if wins   else 0.0
            avg_pct_per_loss  = float(np.mean(losses))  * 100 if losses else 0.0

            top_results.append({
                "strategy_name": name,
                "sharpe": sharpe,
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "num_trades": n_signals,
                "total_pnl": total_pnl,
                "final_capital": eq[-1] if len(eq) > 1 else initial_capital,
                "avg_pct_per_trade": avg_pct_per_trade,
                "avg_pct_per_win":   avg_pct_per_win,
                "avg_pct_per_loss":  avg_pct_per_loss,
            })

        return top_results

    def _compute_shadow_performance(self) -> Dict[str, Dict]:
        """Evaluate every recorded signal against actual next-bar OHLCV.

        Returns a dict keyed by strategy_name with:
          pnl_list  – list of per-signal pnl_pct values
          sharpe    – annualised Sharpe across all signals
        """
        by_strategy: Dict[str, List[float]] = defaultdict(list)
        for record in self._shadow_signals:
            # Unpack with backward compat: entry field added later
            if len(record) == 7:
                date, symbol, sname, direction, stop, target, sig_entry = record
            else:
                date, symbol, sname, direction, stop, target = record
                sig_entry = None

            full_df = self._data_dict.get(symbol)
            if full_df is None:
                continue
            future = full_df[full_df.index > date]
            if future.empty:
                continue
            nb = future.iloc[0]
            entry = float(nb["open"])
            high = float(nb["high"])
            low = float(nb["low"])
            close = float(nb["close"])
            if entry <= 0:
                continue

            # Convert stop/target from absolute prices to % offsets relative to
            # sig_entry, then re-anchor to the actual next-bar open.  This keeps
            # the evaluation correct even when a strategy (e.g. cross_asset_rank)
            # sets stop/target in a different asset's price units.
            ref = float(sig_entry) if sig_entry and sig_entry > 0 else entry
            stop_price  = entry * (1.0 + (stop  - ref) / ref)
            target_price = entry * (1.0 + (target - ref) / ref)

            if direction == Direction.LONG:
                if low <= stop_price:
                    exit_p = stop_price
                elif high >= target_price:
                    exit_p = target_price
                else:
                    exit_p = close
                pnl_pct = (exit_p - entry) / entry
            else:
                if high >= stop_price:
                    exit_p = stop_price
                elif low <= target_price:
                    exit_p = target_price
                else:
                    exit_p = close
                pnl_pct = (entry - exit_p) / entry

            by_strategy[sname].append(pnl_pct)

        result = {}
        for sname, pnl_list in by_strategy.items():
            n = len(pnl_list)
            sharpe = _safe_sharpe(np.array(pnl_list), np.sqrt(252))
            result[sname] = {"pnl_list": pnl_list, "sharpe": sharpe, "signal_count": n}
        return result

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

        # Strategy performance breakdown - computed from ALL trades for that strategy
        # so that Sharpe, win rate, and return are always internally consistent.
        by_strategy: Dict[str, List[Trade]] = defaultdict(list)
        for t in self.all_trades:
            by_strategy[t.strategy_name].append(t)

        strategy_stats = {}
        strategy_sharpes: List[Tuple[str, float]] = []
        for sname, strades in by_strategy.items():
            strades_sorted = sorted(strades, key=lambda t: t.entry_date)
            spnls = [t.pnl for t in strades_sorted]
            swins  = [p for p in spnls if p > 0]
            slosses = [p for p in spnls if p <= 0]
            srets = np.array([t.pnl_pct for t in strades_sorted])
            span = max((strades_sorted[-1].exit_date - strades_sorted[0].entry_date).days, 1)
            tpy = len(strades_sorted) * 365.0 / span
            ssharpe = _safe_sharpe(srets, np.sqrt(tpy))
            pf = (abs(sum(swins)) / abs(sum(slosses))
                  if slosses and sum(slosses) != 0 else float("inf"))
            strategy_stats[sname] = {
                "sharpe": ssharpe,
                "win_rate": len(swins) / len(spnls) if spnls else 0.0,
                "profit_factor": pf,
                "total_pnl": sum(spnls),
                "num_trades": len(strades_sorted),
            }
            strategy_sharpes.append((sname, ssharpe))

        strategy_sharpes.sort(key=lambda x: x[1], reverse=True)
        actual_ranked = strategy_sharpes
        best_strategy = actual_ranked[0][0] if actual_ranked else None
        worst_strategy = actual_ranked[-1][0] if actual_ranked else None

        # Shadow rankings: all strategies that generated at least one signal,
        # evaluated against actual next-bar OHLCV independent of competition.
        shadow_perf = self._compute_shadow_performance()
        shadow_ranked = sorted(
            [(sname, d["sharpe"]) for sname, d in shadow_perf.items()],
            key=lambda x: x[1], reverse=True,
        )

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
            "strategy_rankings": shadow_ranked,   # primary: all strategies with signals
            "actual_trade_rankings": actual_ranked,  # secondary: only strategies that won competition
            "shadow_performance": shadow_perf,
            "daily_logs": self.daily_logs,
            "no_prediction": self.config.no_prediction,
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

def _fmt(v: float, decimals: int = 2) -> str:
    """Format a float with space-thousands separator, e.g. 1 234 567.89"""
    fmt = f"{v:,.{decimals}f}"
    return fmt.replace(",", " ")   # thin space as thousands separator


def print_results(results: Dict, top_strategy_results: Optional[List[Dict]] = None):
    """Pretty-print backtest results."""
    s = results["summary"]
    W = 68
    print("=" * W)
    print("KAIROS BACKTEST RESULTS")
    if results.get("no_prediction"):
        print("  Mode: NO-PREDICTION  (oracle - actual next-bar OHLCV)")
    print("=" * W)
    print(f"  Total Return:     {_fmt(s['total_return']*100)}%")
    print(f"  Sharpe Ratio:     {_fmt(s['sharpe'])}")
    print(f"  Max Drawdown:     {_fmt(s['max_drawdown']*100)}%")
    print(f"  Win Rate:         {_fmt(s['win_rate']*100)}%")
    print(f"  Profit Factor:    {_fmt(s['profit_factor'])}")
    print(f"  Num Trades:       {s['num_trades']:,}".replace(",", " "))
    print(f"  Avg Trade PnL:    {_fmt(s['avg_trade'])}")
    print(f"  Avg Win:          {_fmt(s['avg_win'])}")
    print(f"  Avg Loss:         {_fmt(s['avg_loss'])}")
    print(f"  Final Capital:    {_fmt(s['final_capital'])}")
    print("-" * W)
    print(f"  Best Strategy:    {results['best_strategy']}")
    print(f"  Worst Strategy:   {results['worst_strategy']}")
    print("=" * W)

    if results["strategy_rankings"]:
        shadow_perf = results.get("shadow_performance", {})
        print("\n  ALL STRATEGIES BY SHARPE  (shadow: each signal vs actual next-bar):")
        for i, (name, sharpe) in enumerate(results["strategy_rankings"], 1):
            n_sig = shadow_perf.get(name, {}).get("signal_count", 0)
            print(f"    {i:2d}. {name:35s}  Sharpe: {_fmt(sharpe)}  ({n_sig} signals)")

    if top_strategy_results:
        n_days = len(results.get("equity_curve", [])) or 1
        n_weeks = max(n_days / 5.0, 1.0)
        SW = 114
        print("\n" + "=" * SW)
        print("  ALL STRATEGIES - SHADOW SIGNAL PERFORMANCE")
        print("=" * SW)
        hdr = (f"  {'Strategy':<35}  {'Sharpe':>7}  {'%/trade':>8}  {'%/win':>7}"
               f"  {'%/loss':>7}  {'MaxDD':>8}  {'WinRate':>8}  {'sig/wk':>6}  {'Signals':>7}")
        print(hdr)
        print("  " + "-" * (SW - 2))
        for r in top_strategy_results:
            sharpeStr = f"  {_fmt(r['sharpe']):>7}"
            if (r['sharpe'] < -100):
                sharpeStr = f"<-100"
            spw = r["num_trades"] / n_weeks
            spw_str = "<1" if spw < 1 else str(int(spw))
            print(
                f"  {r['strategy_name']:<35}"
                +sharpeStr+
                f"  {_fmt(r['avg_pct_per_trade']):>7}%"
                f"  {_fmt(r['avg_pct_per_win']):>6}%"
                f"  {_fmt(r['avg_pct_per_loss']):>6}%"
                f"  {_fmt(r['max_drawdown']*100):>7}%"
                f"  {_fmt(r['win_rate']*100):>7}%"
                f"  {spw_str:>6}"
                f"  {r['num_trades']:>7,}".replace(",", " ")
            )
        print("=" * SW)


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
