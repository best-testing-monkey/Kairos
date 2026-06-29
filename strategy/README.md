# Kairos Trading Framework
## Complete Distribution-Based Backtesting System for Crypto

---

## Overview

The Kairos framework is a complete backtesting and live trading system built around a single insight: **if you can predict tomorrow's OHLC as a distribution (60 Monte Carlo samples), you can extract far more alpha than from a point estimate alone.**

The framework contains **42 strategies** across **6 modules**, unified by a master orchestrator that handles multi-asset prediction, online performance tracking, meta-filtering, and execution planning.

---

## Architecture

```
KairosOrchestrator (master controller)
|
|-- MultiAssetKairosPredictor (predicts across BTC/ETH/SOL/...)
|   |-- predict_kairos_cloud(df, signal="BTC-USD") -> 60 samples
|
|-- StrategyRegistry (42 strategies)
|   |-- Base (18) | Path (5) | Horizon (3) | Execution (7) | Meta (9)
|
|-- DecisionTreeRouter (entropy -> bimodality -> regime -> EV -> size)
|
|-- StrategyPerformanceTracker (rolling 30-trade Sharpe per strategy)
|
|-- Meta-filters (kurtosis, volume, bimodality)
|
|-- ExecutionEngine (partial exits, multi-day holds, trailing stops)
|
|-- Results (equity curve, trade log, strategy rankings)
```

---

## Modules

### 1. kairos_backtest.py (18 strategies)
Core distribution analysis and base strategies.

| Strategy | Description |
|----------|-------------|
| PercentileEntry | Long at 15th pct, short at 85th pct of predicted close |
| DynamicBracket | Hard stops at 10th/90th pct, sized by inverse variance |
| Skew | Trade in direction of distribution skew |
| RangeTrading | Mean reversion when predicted range is tight |
| TrendFollowing | Breakout when predicted close is far and std is low |
| VolatilityArb | Compare predicted realized vol to implied vol |
| HighLow | Trade predicted high/low directly |
| OpenGap | Trade gap between predicted open and current close |
| FadeExtreme | Fade moves near predicted extreme in tight range |
| MomentumContinuation | High-conviction breakout when CV is low |
| ExpectedValue | Pure EV maximization with ATR brackets |
| MartingaleFloor | Scale in near predicted low with statistical floor |
| RSIFilter | Only take signals when RSI confirms |
| MACDFilter | Only take signals when MACD crossover aligns |
| BollingerValidation | Fade breakouts when predicted high is inside BB |
| SupportConfluence | Enter when predicted low aligns with VW support |
| InverseVariance | Size purely by 1/sigma^2 |
| CloseDirection | Simple directional bet sized by predicted Sharpe |

### 2. kairos_path.py (5 strategies)
Extracts intra-day path patterns from 60-sample ordering.

| Strategy | Description |
|----------|-------------|
| PathRally | Long when trajectory = rally, low-before-high > 0.6 |
| PathFade | Short when trajectory = fade, high-before-low > 0.6 |
| PathVShape | Buy the dip at predicted low, target predicted close |
| PathInvertedV | Short the pop at predicted high, target predicted close |
| PathHighLowSequence | Pure sequence bet: buy if low first, short if high first |

### 3. kairos_horizon.py (3 strategies)
Multi-horizon prediction stack for hold-period optimization.

| Strategy | Description |
|----------|-------------|
| MultiHorizonHold | Enters T+1, holds for recommended duration (1-3 days) |
| ConfidenceDecayFilter | Only enters if std decay curve is sub-linear |
| RollingHorizon | Daily re-evaluation; reduces size if T+2 no longer confirms |

### 4. kairos_execution.py (7 strategies)
Path-dependent partial exits and volume/amount integration.

| Strategy | Description |
|----------|-------------|
| PathExecution | 3-leg scale-out: 33% at high, 33% at close, 34% trailing |
| PathHighLowExecution | 2-leg: 50% at predicted high, 50% at predicted close |
| VolumeConfirmation | High volume confirms move; low volume fades it |
| VolumeFade | Pure contrarian: fade moves on declining predicted volume |
| AmountFlow | Uses predicted notional flow as directional signal |
| PredictedVWAP | Close > predicted VWAP = bullish; < = bearish |
| LiquidityFilter | Wrapper: only passes if predicted volume > historical pct |

### 5. kairos_meta.py (9 strategies)
Cross-asset ranking, online performance tracking, and tail trading.

| Strategy | Description |
|----------|-------------|
| CrossAssetRank | Allocates 100% to top predicted Sharpe asset |
| CrossAssetSpread | Pairs trade: long best, short worst |
| CrossAssetMomentumTransfer | Front-run lagging assets using leader's signal |
| OnlineWeighted | Weights all strategies by rolling Sharpe, picks best |
| ThompsonSampling | Beta-distributed strategy selection (explore/exploit) |
| RegimeSwitching | Switches strategy sets based on detected regime |
| TailAsymmetry | Trades left vs right tail fatness |
| BuyWings | Buys OTM options when kurtosis is high and IV is cheap |
| SellPremium | Sells straddles when kurtosis is low and IV > realized |

### 6. kairos_orchestrator.py (master controller)
Wires everything together.

---

## Quick Start

### Single Asset Backtest

```python
from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig, print_results

config = OrchestratorConfig(
    initial_capital=10000.0,
    fee_pct=0.001,
    slippage_pct=0.0005,
    cross_asset_ranking=False,  # Single asset
    online_weighting=True,
    partial_exits=True,
    max_horizon=3,
)

orchestrator = KairosOrchestrator(
    predict_fn=predict_kairos_cloud,
    assets=["BTC-USD"],
    config=config,
)

results = orchestrator.run_single_asset(btc_df, lookback=200)
print_results(results)
```

### Multi-Asset Backtest

```python
config = OrchestratorConfig(
    cross_asset_ranking=True,
    max_positions_per_day=2,
)

orchestrator = KairosOrchestrator(
    predict_fn=predict_kairos_cloud,
    assets=["BTC-USD", "ETH-USD", "SOL-USD"],
    config=config,
)

results = orchestrator.run_backtest({
    "BTC-USD": btc_df,
    "ETH-USD": eth_df,
    "SOL-USD": sol_df,
}, lookback=200)

print_results(results)
```

### Live Signal

```python
histories = {
    "BTC-USD": btc_df,  # Up to current bar
    "ETH-USD": eth_df,
}

signal = orchestrator.get_live_signal(histories)
if signal:
    print(f"Enter {signal.symbol} {signal.direction} at {signal.entry_price}")
    print(f"Stop: {signal.stop_price}, Target: {signal.target_price}")
    print(f"Hold: {signal.hold_days} days, Size: {signal.size:.2%}")
```

---

## The predict_kairos_cloud Contract

Your predictor must implement this signature:

```python
def predict_kairos_cloud(df: pd.DataFrame, signal: str = "BTC-USD") -> List[pd.DataFrame]:
    """
    Args:
        df: History DataFrame with columns [open, high, low, close, volume, amount]
            Index is DatetimeIndex.
        signal: Asset symbol (e.g., "BTC-USD", "ETH-USD")

    Returns:
        List of 60 single-row DataFrames, each with columns:
        [open, high, low, close, volume, amount]
        Each row represents one Monte Carlo sample of the next bar.
    """
    pass
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 10000.0 | Starting capital |
| `fee_pct` | 0.001 | Per-trade fee (0.1%) |
| `slippage_pct` | 0.0005 | Slippage (0.05%) |
| `entropy_threshold` | 3.0 | Max entropy to allow trade |
| `bimodality_filter` | True | Block bimodal distributions |
| `kurtosis_max` | 3.0 | Max kurtosis for directional trades |
| `kurtosis_action` | "block" | "block", "reduce", or "invert" |
| `min_volume_percentile` | 30.0 | Min predicted volume vs history |
| `performance_lookback` | 30 | Rolling trade window for Sharpe |
| `online_weighting` | True | Weight strategies by recent Sharpe |
| `thompson_sampling` | False | Use Thompson sampling instead |
| `temperature` | 0.5 | Softmax temperature for weighting |
| `cross_asset_ranking` | True | Rank assets by predicted Sharpe |
| `max_positions_per_day` | 3 | Max new positions per day |
| `max_position_size_pct` | 0.5 | Max 50% of capital per trade |
| `max_total_exposure` | 1.0 | Max 100% of capital deployed |
| `partial_exits` | True | Enable 3-leg partial exits |
| `max_horizon` | 3 | Max prediction horizon (days) |
| `multi_horizon_holds` | True | Enable multi-day holds |

---

## Decision Tree

```
Kairos Distribution (60 samples)
|
|-- [1] ENTROPY FILTER
|   High entropy (> threshold) --> NO TRADE
|
|-- [2] BIMODALITY FILTER
|   Bimodal --> NO TRADE
|
|-- [3] KURTOSIS FILTER
|   High kurtosis + action="block" --> NO TRADE
|   High kurtosis + action="reduce" --> HALVE SIZE
|   High kurtosis + action="invert" --> FLIP DIRECTION
|
|-- [4] VOLUME FILTER
|   Predicted volume < historical percentile --> NO TRADE
|
|-- [5] REGIME DETECTION
|   RANGE --> RangeTrading, PercentileEntry, FadeExtreme, PathVShape, etc.
|   TREND --> TrendFollowing, Skew, Momentum, PathRally, etc.
|   UNCERTAIN --> ExpectedValue, InverseVariance
|
|-- [6] STRATEGY EVALUATION (all 42)
|   Each strategy generates a Signal
|
|-- [7] ONLINE WEIGHTING
|   Weight each Signal by rolling Sharpe of its strategy
|
|-- [8] CROSS-ASSET RANKING (if multi-asset)
|   Pick top N assets by weighted expected value
|
|-- [9] EXPECTED VALUE FILTER
|   EV <= 0 --> NO TRADE
|
|-- [10] POSITION SIZING
|    Kelly Criterion (half-Kelly cap) + hold penalty + online weight
|
|-- [11] EXECUTION PLAN
|    Build partial exit legs (target, stop, trailing stop)
```

---

## Output Format

The `run_backtest()` method returns a dict with:

```python
{
    "summary": {
        "total_return": 0.45,      # 45%
        "sharpe": 1.82,
        "max_drawdown": 0.12,      # 12%
        "win_rate": 0.58,          # 58%
        "profit_factor": 2.1,
        "num_trades": 342,
        "avg_trade": 125.50,
        "avg_win": 450.20,
        "avg_loss": -180.30,
        "final_capital": 14500.00,
    },
    "equity_curve": [(date, capital), ...],
    "trades": [Trade, ...],
    "signals": [UnifiedSignal, ...],
    "strategy_performance": {
        "high_low": {"sharpe": 2.1, "win_rate": 0.62, ...},
        ...
    },
    "best_strategy": "high_low",
    "worst_strategy": "macd_filter",
    "strategy_rankings": [(name, sharpe), ...],
    "daily_logs": [{date, capital, num_signals, num_positions}, ...],
}
```

---

## Dependencies

- pandas >= 1.3
- numpy >= 1.20
- scipy >= 1.7

No external trading libraries. No GPU required.

---

## Files

| File | Description | Strategies |
|------|-------------|------------|
| `kairos_backtest.py` | Core distribution, base strategies, standard engine | 18 |
| `kairos_path.py` | Path extraction, path-aware strategies | 5 |
| `kairos_horizon.py` | Multi-horizon stack, hold strategies | 3 |
| `kairos_execution.py` | Partial exits, volume integration | 7 |
| `kairos_meta.py` | Cross-asset, online tracking, tail trading | 9 |
| `kairos_orchestrator.py` | Master controller, unified API | 0 (wires all) |

**Total: 42 strategies + 1 orchestrator**

---

## License

Use at your own risk. This is research code, not production trading software.
