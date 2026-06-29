# Kairos Framework: Missing Strategies Design Document

**Version:** 1.0  
**Date:** 2026-06-29  
**Target:** Kimi Code implementation  
**Scope:** 12 strategies/features requiring no external data feeds

---

## 1. Overview

This document specifies the implementation of strategies and features discussed in the Kairos framework design but not yet coded. All implementations must:

1. Import from existing modules (`kairos_backtest.py`, `kairos_path.py`, `kairos_horizon.py`, `kairos_execution.py`, `kairos_meta.py`)
2. Inherit from `Strategy` base class
3. Return `Signal` objects or `None`
4. Use only data available from the 60-sample Kairos distribution and historical OHLCV
5. Be dependency-light (pandas, numpy, scipy only)

---

## 2. Strategy Specifications

### 2.1 VaR Position Cap

**ID:** `var_position_cap`  
**Module:** `kairos_backtest.py`  
**Type:** Sizing modifier (wraps base strategy)

**Description:**  
Calculate the 5th percentile of the predicted close distribution. Size the position so that if the 5th percentile hits, the account loses no more than a configurable percentage (default 1%).

**Algorithm:**
```
1. Let base_signal = base_strategy.generate_signal(...)
2. If base_signal is None: return None
3. Let var_5 = dist.stats["close"]["pct_5"]
4. Let entry = base_signal.entry
5. If LONG: max_loss_per_unit = entry - var_5
   If SHORT: max_loss_per_unit = var_5 - entry
6. If max_loss_per_unit <= 0: return base_signal (VaR is favorable)
7. account_risk_limit = capital * max_account_risk_pct
8. max_units = account_risk_limit / max_loss_per_unit
9. max_notional = max_units * entry
10. max_size = max_notional / capital
11. base_signal.size = min(base_signal.size, max_size)
12. Return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy, max_account_risk_pct: float = 0.01)
```

**Acceptance Criteria:**
- When predicted 5th percentile is 2% below entry, max size = 0.5 (for 1% risk limit)
- When predicted 5th percentile is above entry (for longs), signal passes through unchanged
- Must never increase size above base strategy's recommendation
- Must handle the case where var_5 equals entry (return None or size=0)

---

### 2.2 Distribution Overlap Classifier

**ID:** `distribution_overlap`  
**Module:** `kairos_backtest.py`  
**Type:** Standalone strategy

**Description:**  
Compute the overlap coefficient between today's predicted distribution and yesterday's. High overlap (>0.85) = range-bound (mean reversion). Low overlap (<0.60) = gap/trend (momentum). Use this to select strategy class automatically.

**Algorithm:**
```
1. Requires yesterday's distribution in context["prev_dist"]
2. If prev_dist is None: return None
3. overlap = dist.overlap_coefficient(prev_dist, col="close")
4. If overlap > range_threshold: use RangeTradingStrategy logic
   If overlap < trend_threshold: use TrendFollowingStrategy logic
   Else: return None
5. Generate signal using selected sub-strategy
```

**Constructor:**
```python
def __init__(self, range_threshold: float = 0.85, trend_threshold: float = 0.60)
```

**Integration Note:**  
The backtest engine must pass `prev_dist` in context. In `KairosOrchestrator._run_day()`, add:
```python
context["prev_dist"] = self._prev_dist.get(symbol)
self._prev_dist[symbol] = dist
```

**Acceptance Criteria:**
- Overlap > 0.85 generates a range-trading signal
- Overlap < 0.60 generates a trend-following signal
- Must not generate signals when overlap is between thresholds
- Must use the `KairosDistribution.overlap_coefficient()` method already implemented

---

### 2.3 Conditional Path Probability

**ID:** `conditional_path`  
**Module:** `kairos_path.py`  
**Type:** Standalone strategy

**Description:**  
From the 60 samples, compute P(hit predicted high AND predicted low). If high probability (>0.70), it's a range day (sell straddle). If low probability (<0.30), it's a trend day (buy directional).

**Algorithm:**
```
1. pred_high = dist.stats["high"]["mean"]
2. pred_low = dist.stats["low"]["mean"]
3. count = 0
4. For each sample in dist.predictions:
   if sample["high"] >= pred_high and sample["low"] <= pred_low:
       count += 1
5. p_range = count / len(dist.predictions)
6. If p_range > range_threshold:
      Return FLAT signal with metadata action="sell_straddle", confidence=p_range
   If p_range < trend_threshold:
      direction = LONG if close_mean > current_price else SHORT
      stop = low_pct_10 (long) or high_pct_90 (short)
      target = high_pct_90 (long) or low_pct_10 (short)
      Return directional signal with confidence=1-p_range
   Else: return None
```

**Constructor:**
```python
def __init__(self, range_threshold: float = 0.70, trend_threshold: float = 0.30)
```

**Acceptance Criteria:**
- p_range > 0.70 returns a FLAT signal with "sell_straddle" metadata
- p_range < 0.30 returns a directional signal
- Must iterate all 60 samples in `dist.predictions`, not just aggregated stats
- Must use the raw sample DataFrames, not the summary statistics

---

### 2.4 Residual Tracking & Model Decay Monitor

**ID:** `model_decay_monitor`  
**Module:** `kairos_backtest.py`  
**Type:** Meta-filter / sizing modifier

**Description:**  
Track what percentage of realized prices fall within the predicted 1σ and 2σ bands. If calibration drifts, dynamically widen stops or reduce size until the model recalibrates.

**Algorithm:**
```
1. Maintain calibration_history: deque of tuples
   (predicted_mean, predicted_std, realized_close, in_1sigma, in_2sigma)

2. After each bar closes:
   in_1sigma = realized_close in [mean - std, mean + std]
   in_2sigma = realized_close in [mean - 2*std, mean + 2*std]
   Append to calibration_history

3. If len(history) >= lookback:
   hit_rate_1s = mean(in_1sigma over last lookback)
   hit_rate_2s = mean(in_2sigma over last lookback)

4. If hit_rate_1s < target_1sigma * 0.8:  # too tight
      calibration_factor = widen_factor
      size_factor = 0.5
   Elif hit_rate_1s > target_1sigma * 1.2:  # too loose
      calibration_factor = tighten_factor
      size_factor = 1.0
   Else:
      calibration_factor = 1.0
      size_factor = 1.0

5. Apply to base strategy signal:
   signal.stop *= calibration_factor
   signal.size *= size_factor
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             lookback: int = 30,
             target_1sigma: float = 0.68,
             target_2sigma: float = 0.95,
             widen_factor: float = 1.5,
             tighten_factor: float = 0.8)
```

**State Management:**
```python
self.calibration_history: deque = deque(maxlen=lookback * 2)
```

**Method to update after each bar:**
```python
def update_calibration(self, dist: KairosDistribution, realized_close: float):
    mean = dist.stats["close"]["mean"]
    std = dist.stats["close"]["std"]
    in_1s = (mean - std) <= realized_close <= (mean + std)
    in_2s = (mean - 2*std) <= realized_close <= (mean + 2*std)
    self.calibration_history.append((mean, std, realized_close, in_1s, in_2s))
```

**Acceptance Criteria:**
- After 30 bars with 50% 1σ hit rate (target=68%), stops widen by 1.5x
- After 30 bars with 80% 1σ hit rate, stops tighten to 0.8x
- Size modifier bounded [0.25, 1.0]
- Must not modify signal if calibration is within tolerance (target +/- 20%)

---

### 2.5 Pyramiding

**ID:** `pyramiding`  
**Module:** `kairos_execution.py`  
**Type:** Position management strategy (extends base strategy)

**Description:**  
Add to position size every time price moves a threshold in your favor and the updated distribution still confirms direction.

**Algorithm:**
```
1. Generate initial signal from base_strategy
2. Track pyramid state per position:
   pyramid_levels: list of (add_price, add_size, stop, target)
   total_size = sum of all levels

3. Each bar, for each active pyramid position:
   a. Compute price_move = (current_price - entry) / entry (for longs)
   b. If price_move >= pyramid_threshold_pct * (level_count + 1):
      i. Re-predict with updated history
      ii. If new_dist confirms original direction:
          - Add new level: size = base_size * pyramid_add_pct
          - New stop = max(old_stop, new_dist.low_pct_10)
          - New target = min(old_target, new_dist.high_pct_90)
          - level_count += 1
      iii. If level_count >= max_pyramid_levels: stop adding

4. Return the base signal with metadata containing pyramid plan
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             pyramid_threshold_pct: float = 0.01,
             pyramid_add_pct: float = 0.25,
             max_pyramid_levels: int = 3)
```

**State:**
```python
self.active_pyramids: Dict[str, List[Dict]] = {}
# Key: symbol, Value: list of pyramid levels
# Each level: {"entry": float, "size": float, "stop": float, "target": float}
```

**Engine Integration:**
- The orchestrator must support "add to position" by treating pyramid adds as separate position entries
- OR: Return a single signal with `metadata["pyramid_plan"]` containing all levels

**Acceptance Criteria:**
- After 1% favorable move + confirmation, a new pyramid level is created
- Maximum 3 pyramid levels total
- Each level has its own stop, but the overall position stop is the tightest of all levels
- Must reduce or skip if updated distribution no longer confirms direction

---

### 2.6 Time-Based Stops

**ID:** `time_based_stop`  
**Module:** `kairos_execution.py`  
**Type:** Exit modifier (wraps base strategy)

**Description:**  
If price hasn't touched predicted high/low by 75% of the session, exit early. For daily crypto bars, this simplifies to: if target not hit by bar close, exit at close.

**Algorithm:**
```
1. Base strategy generates signal
2. Add metadata:
   - "time_exit_enabled": True
   - "time_exit_price": predicted close median (pct_50)
   - "time_exit_bar": current_bar + time_bars

3. In the engine, when managing positions:
   If position has time_exit_enabled:
      If current_bar >= time_exit_bar and target not yet hit:
         Exit at close
         Reason: "time_stop"
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             time_bars: int = 1,  # For daily data, exit after 1 bar if target not hit
             exit_at: str = "close")  # "close" or "predicted_median"
```

**Engine Integration:**
- The orchestrator must check `signal.metadata.get("time_exit_enabled")` when creating positions
- Add `time_exit_bar` to the position dict
- In `_manage_positions()`, check if current bar >= time_exit_bar before checking stop/target

**Acceptance Criteria:**
- If target not reached by time_exit_bar, position exits at that bar's close
- If target IS reached before time_exit_bar, normal management continues
- Stop-loss always takes precedence over time-based exit
- For daily bars, time_bars=1 means exit at close if not hit

---

### 2.7 Regime Clustering (KNN Strategy Selector)

**ID:** `regime_cluster`  
**Module:** `kairos_meta.py`  
**Type:** Meta-strategy

**Description:**  
Cluster the last N distributions by feature vector. Find K nearest neighbors. Run the strategy that performed best in that neighborhood.

**Feature Vector (5 dimensions):**
```python
features = [
    dist.coefficient_of_variation("close"),      # [0, inf]
    dist.stats["close"]["skew"],                  # [-inf, inf]
    (dist.stats["close"]["pct_90"] - dist.stats["close"]["pct_10"]) / current_price,  # [0, inf]
    dist.entropy("close"),                         # [0, inf]
    1.0 if dist.stats["close"]["mean"] > current_price else -1.0,  # {-1, 1}
]
```

**Algorithm:**
```
1. Maintain feature_buffer: deque of (features, strategy_name, pnl)

2. For new prediction:
   a. Compute feature vector f_new
   b. Normalize all features in buffer to [0, 1] using min/max
   c. Compute Euclidean distance from f_new to all buffer entries
   d. Select K nearest neighbors
   e. If nearest distance > distance_threshold: fall back to OnlineWeightedStrategy
   f. Group neighbors by strategy_name
   g. Pick strategy with highest average PnL among neighbors
   h. Run that strategy

3. After trade completes, append (features, strategy_name, pnl) to buffer
```

**Constructor:**
```python
def __init__(self, base_strategies: List[Strategy],
             feature_buffer_size: int = 100,
             k_neighbors: int = 5,
             distance_threshold: float = 0.5,
             fallback_strategy: Optional[Strategy] = None)
```

**Normalization:**
```python
def _normalize_features(self, features: List[float], buffer: List) -> List[float]:
    normalized = []
    for i in range(5):
        vals = [entry[0][i] for entry in buffer]
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            normalized.append(0.5)
        else:
            normalized.append((features[i] - min_v) / (max_v - min_v))
    return normalized
```

**Acceptance Criteria:**
- After 50 trades in buffer, selects strategy based on KNN performance
- Falls back to `fallback_strategy` (or first base strategy) if no close neighbors
- Feature buffer persists across the entire backtest
- Must handle empty buffer gracefully (return first base strategy)

---

### 2.8 Overnight Exposure Filter

**ID:** `overnight_filter`  
**Module:** `kairos_backtest.py`  
**Type:** Filter (wraps base strategy)

**Description:**  
If predicted next-day range is entirely below your entry, close the position before overnight. If entirely above, hold or add.

**Algorithm:**
```
1. Requires context["current_position"] = position dict or None
2. If no current position: pass through to base strategy
3. If current position exists:
   a. pred_high = dist.stats["high"]["mean"]
   b. pred_low = dist.stats["low"]["mean"]
   c. If LONG and pred_high < entry_price:
      - Return FLAT signal
      - metadata: action="close_overnight", reason="range_below_entry"
   d. If SHORT and pred_low > entry_price:
      - Return FLAT signal
      - metadata: action="close_overnight", reason="range_above_entry"
   e. Else: pass through to base strategy
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy)
```

**Engine Integration:**
- Engine must pass `current_position` in context
- FLAT signals with `action="close_overnight"` should trigger immediate position close
- In `KairosOrchestrator`, add to context before signal generation:
```python
context["current_position"] = next(
    (p for p in self.active_positions if p["symbol"] == symbol), None
)
```

**Acceptance Criteria:**
- Long position with predicted high < entry -> FLAT signal generated
- Short position with predicted low > entry -> FLAT signal generated
- All other cases pass through to base strategy unchanged
- Must not interfere with normal entry signals (only modifies existing positions)

---

### 2.9 RSI Divergence Confirmation

**ID:** `rsi_divergence`  
**Module:** `kairos_backtest.py`  
**Type:** Standalone strategy

**Description:**  
Detect bearish RSI divergence (price makes higher high, RSI makes lower high) and confirm with Kairos predicted lower low. Or bullish divergence (price lower low, RSI higher low) confirmed by predicted higher high.

**Algorithm:**
```
1. Compute RSI over lookback period
2. Find local price highs/lows in the lookback window
3. Find local RSI highs/lows at corresponding bars
4. Check for divergence:
   Bullish: price_low2 < price_low1 AND rsi_low2 > rsi_low1
   Bearish: price_high2 > price_high1 AND rsi_high2 < rsi_high1

5. If divergence detected:
   a. Check Kairos prediction:
      - Bullish divergence + predicted close > current_price -> LONG
      - Bearish divergence + predicted close < current_price -> SHORT
   b. If Kairos does NOT confirm: return None (divergence is false)

6. Set stops and targets from distribution
```

**Constructor:**
```python
def __init__(self, rsi_period: int = 14,
             lookback_bars: int = 20,
             divergence_threshold: float = 2.0)  # Min RSI point difference
```

**RSI Calculation:**
Reuse existing `_rsi()` method from `RSIFilterStrategy` or implement standalone.

**Local Extrema Detection:**
```python
def _find_local_extrema(self, series: np.ndarray, order: int = 2) -> Tuple[List[int], List[int]]:
    from scipy.signal import argrelextrema
    highs = argrelextrema(series, np.greater, order=order)[0]
    lows = argrelextrema(series, np.less, order=order)[0]
    return highs, lows
```

**Acceptance Criteria:**
- Detects at least one bullish and one bearish divergence in a test dataset
- Only generates signal if Kairos prediction confirms the divergence direction
- Returns None if divergence exists but Kairos contradicts it
- Must use at least 2 pivot points for divergence detection

---

### 2.10 Leverage Calibration

**ID:** `leverage_calibration`  
**Module:** `kairos_backtest.py`  
**Type:** Sizing modifier (wraps base strategy)

**Description:**  
Predicted range is +/-1% -> use 5x leverage. Predicted range is +/-8% -> use 1x or spot. Size leverage to predicted volatility instead of using fixed leverage.

**Algorithm:**
```
1. Let base_signal = base_strategy.generate_signal(...)
2. If base_signal is None: return None
3. pred_range_pct = (dist.stats["close"]["pct_90"] - dist.stats["close"]["pct_10"]) / current_price
4. leverage_map:
   - If pred_range_pct < 0.02: leverage = 5.0
   - If pred_range_pct < 0.04: leverage = 3.0
   - If pred_range_pct < 0.06: leverage = 2.0
   - Else: leverage = 1.0
5. base_signal.size *= leverage
6. Cap at max_leverage and max_position_size_pct from config
7. Return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             leverage_tiers: Optional[List[Tuple[float, float]]] = None,
             max_leverage: float = 5.0)
```

**Default Tiers:**
```python
[(0.02, 5.0), (0.04, 3.0), (0.06, 2.0), (float("inf"), 1.0)]
# Format: (max_range_pct, leverage)
```

**Acceptance Criteria:**
- Predicted range < 2% -> leverage = 5x
- Predicted range 2-4% -> leverage = 3x
- Predicted range 4-6% -> leverage = 2x
- Predicted range > 6% -> leverage = 1x
- Must cap final size at max_leverage and respect account limits
- Must not apply leverage to FLAT signals

---

### 2.11 Monte Carlo Scenario Planning

**ID:** `monte_carlo_scenario`  
**Module:** `kairos_meta.py`  
**Type:** Strategy evaluator (meta-strategy)

**Description:**  
Use the mean and sigma from the 60-sample distribution to generate 1,000 synthetic paths. Test each candidate strategy against all paths. Pick the one with the highest expected return across the ensemble.

**Algorithm:**
```
1. Extract mean and std from dist.stats["close"]
2. Generate 1,000 synthetic next-day closes:
   synthetic_closes = np.random.normal(mean, std, 1000)
3. For each base_strategy:
   a. Simulate the strategy against each synthetic close
   b. Compute PnL for each path
   c. Compute expected PnL and Sharpe across all paths
4. Select strategy with highest expected PnL
5. Return that strategy's signal
```

**Constructor:**
```python
def __init__(self, base_strategies: List[Strategy],
             n_scenarios: int = 1000,
             selection_metric: str = "expected_pnl")  # "expected_pnl" or "sharpe"
```

**Simulation Logic:**
```python
def _simulate_strategy(self, strategy: Strategy, dist: KairosDistribution,
                       current_price: float, history: pd.DataFrame,
                       scenarios: np.ndarray) -> Tuple[float, float]:
    pnls = []
    for close in scenarios:
        sig = strategy.generate_signal(dist, current_price, history, {})
        if sig is None or sig.direction == Direction.FLAT:
            pnls.append(0.0)
            continue
        pnl = (close - sig.entry) * sig.direction.value * sig.size
        pnls.append(pnl)

    expected_pnl = np.mean(pnls)
    sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0.0
    return expected_pnl, sharpe
```

**Acceptance Criteria:**
- Generates 1,000 synthetic closes from N(mean, std)
- Evaluates all base strategies against the ensemble
- Selects strategy with highest mean PnL (or Sharpe)
- Returns the selected strategy's actual signal (not the synthetic one)
- Must be computationally efficient (vectorized where possible)

---

### 2.12 Predicted Path Integration (Multi-Day Trajectory)

**ID:** `path_integration`  
**Module:** `kairos_horizon.py`  
**Type:** Multi-horizon strategy

**Description:**  
Build a synthetic "most likely path" from the sequence of T+1, T+2, T+3 distributions. If the path shows a 3-day rally with tightening variance, hold for 3 days. If it shows a spike in entropy on day 2, exit before day 2.

**Algorithm:**
```
1. Requires HorizonStack with horizons 1, 2, 3
2. Extract median path points:
   day1 = stack.horizons[1].stats["close"]["mean"]
   day2 = stack.horizons[2].stats["close"]["mean"]
   day3 = stack.horizons[3].stats["close"]["mean"]

3. Compute path characteristics:
   - direction_consistency: all three days agree on direction?
   - variance_trend: std decreasing (tightening) or increasing?
   - entropy_trend: entropy decreasing (more certain) or increasing?

4. Decision rules:
   If direction_consistent AND variance_tightening:
      hold_days = 3, confidence = high
   If direction_consistent AND variance_flat:
      hold_days = 2, confidence = medium
   If direction_inconsistent OR entropy_spike:
      hold_days = 1, confidence = low
   If day2_mean reverses direction from day1:
      hold_days = 1, exit_before_day2 = True

5. Generate signal with computed hold_days
```

**Constructor:**
```python
def __init__(self, max_horizon: int = 3,
             variance_tightening_threshold: float = 0.9,  # std_day2 < 0.9 * std_day1
             entropy_spike_threshold: float = 1.5)
```

**Path Quality Score:**
```python
def _path_quality(self, stack: HorizonStack) -> Tuple[int, float]:
    h1 = stack.horizons[1].stats["close"]
    h2 = stack.horizons[2].stats["close"] if 2 in stack.horizons else h1
    h3 = stack.horizons[3].stats["close"] if 3 in stack.horizons else h2

    d1 = 1 if h1["mean"] > stack.base_price else -1
    d2 = 1 if h2["mean"] > stack.base_price else -1
    d3 = 1 if h3["mean"] > stack.base_price else -1

    consistent = (d1 == d2 == d3)
    tightening = h2["std"] < h1["std"] * 0.9 and h3["std"] < h2["std"] * 0.9

    if consistent and tightening:
        return 3, 0.9
    elif consistent:
        return 2, 0.7
    elif d1 == d2 and d2 != d3:
        return 2, 0.5  # Exit before day 3
    else:
        return 1, 0.3
```

**Acceptance Criteria:**
- 3-day consistent rally with tightening variance -> hold_3_days
- 2-day consistent then reversal -> hold_2_days
- Day 2 entropy spike -> hold_1_day
- Must use the HorizonStack already computed by KairosMultiHorizonPredictor

---

## 3. Engine Modifications

### 3.1 Orchestrator Context Updates

In `KairosOrchestrator._run_day()`, add to context:
```python
context["prev_dist"] = self._prev_dist.get(symbol)
context["current_position"] = next(
    (p for p in self.active_positions if p["symbol"] == symbol), None
)
# After prediction:
self._prev_dist[symbol] = dist
```

Add to `__init__`:
```python
self._prev_dist: Dict[str, KairosDistribution] = {}
```

### 3.2 Position Management Updates

In `_manage_positions()`, add time-based exit check:
```python
# After stop/target checks, before hold expiry:
if exit_price is None and pos.get("time_exit_bar") is not None:
    if current_bar_index >= pos["time_exit_bar"]:
        exit_price = close
        exit_reason = "time_stop"
```

### 3.3 Calibration Update Hook

Add after each bar close in `_run_day()`:
```python
# Update model decay monitor if any strategy uses it
for strat in self.strategies:
    if hasattr(strat, "update_calibration"):
        realized = float(tomorrow["close"])
        strat.update_calibration(dist, realized)
```

### 3.4 Pyramid Position Support

In `_enter_position()`, handle pyramid adds:
```python
if signal.metadata.get("pyramid_add"):
    # Find existing position for this symbol
    existing = next((p for p in self.active_positions if p["symbol"] == symbol), None)
    if existing:
        # Add to existing position
        existing["size"] += new_size
        existing["stop"] = max(existing["stop"], new_stop)  # Tightest stop
        existing["target"] = min(existing["target"], new_target)  # Closest target
    else:
        # New position
        ...
```

---

## 4. Testing Plan

### 4.1 Unit Tests (per strategy)

For each strategy, create a test with synthetic data:

```python
def test_var_position_cap():
    # Create distribution where 5th percentile is 2% below current price
    # Size should be capped at 0.5 for 1% risk limit
    pass

def test_distribution_overlap():
    # Create two distributions with overlap > 0.85
    # Should return range-trading signal
    pass

def test_conditional_path():
    # Create 60 samples where 80% hit both high and low
    # Should return FLAT with "sell_straddle" metadata
    pass
```

### 4.2 Integration Tests

```python
def test_orchestrator_with_all_strategies():
    # Run backtest with all 54 strategies (42 existing + 12 new)
    # Verify no exceptions
    # Verify results dict has all expected keys
    pass

def test_cross_asset_with_new_strategies():
    # Run multi-asset backtest including CrossAssetRank + VaR cap
    # Verify position sizes respect VaR limits
    pass
```

### 4.3 Acceptance Criteria Summary

| Strategy | Test | Pass Criteria |
|----------|------|---------------|
| VaR Cap | Size with 2% VaR, 1% limit | size <= 0.5 |
| Overlap | Overlap=0.9 | Range signal generated |
| Conditional Path | 80% range samples | FLAT + metadata |
| Model Decay | 50% 1σ hit rate | Stops 1.5x wider |
| Pyramiding | 1% move + confirmation | 2 levels created |
| Time Stop | Target not hit | Exit at close |
| Regime Cluster | 50 trades in buffer | KNN selection active |
| Overnight | Pred high < entry | FLAT signal |
| RSI Divergence | Bullish div + Kairos up | LONG signal |
| Leverage | Range < 2% | 5x size multiplier |
| Monte Carlo | 1000 scenarios | Strategy selected |
| Path Integration | 3-day tightening | hold_days=3 |

---

## 5. Implementation Order

**Phase 1 (No engine changes):**
1. VaR Position Cap
2. Conditional Path Probability
3. Leverage Calibration
4. RSI Divergence Confirmation
5. Monte Carlo Scenario Planning

**Phase 2 (Minor engine changes):**
6. Distribution Overlap Classifier (add prev_dist to context)
7. Time-Based Stops (add time_exit_bar to position)
8. Overnight Exposure Filter (add current_position to context)

**Phase 3 (State management):**
9. Model Decay Monitor (add calibration_history)
10. Pyramiding (add pyramid state)
11. Regime Clustering (add feature_buffer)

**Phase 4 (Integration):**
12. Predicted Path Integration (uses HorizonStack)

---

## 6. File Placement

| Strategy | File | Notes |
|----------|------|-------|
| VaR Position Cap | `kairos_backtest.py` | Sizing wrapper |
| Distribution Overlap | `kairos_backtest.py` | Standalone |
| Conditional Path | `kairos_path.py` | Uses raw samples |
| Model Decay | `kairos_backtest.py` | Stateful wrapper |
| Pyramiding | `kairos_execution.py` | Position manager |
| Time-Based Stops | `kairos_execution.py` | Exit wrapper |
| Regime Clustering | `kairos_meta.py` | KNN meta-strategy |
| Overnight Filter | `kairos_backtest.py` | Filter wrapper |
| RSI Divergence | `kairos_backtest.py` | Standalone |
| Leverage Calibration | `kairos_backtest.py` | Sizing wrapper |
| Monte Carlo | `kairos_meta.py` | Meta-strategy |
| Path Integration | `kairos_horizon.py` | Multi-horizon |

---

## 7. Post-Implementation Checklist

- [ ] All 12 strategies inherit from `Strategy`
- [ ] All 12 strategies return `Signal` or `None`
- [ ] All 12 strategies have unique `name` attributes
- [ ] `StrategyRegistry.build_all()` includes all 12 new strategies
- [ ] `KairosOrchestrator` context includes `prev_dist` and `current_position`
- [ ] Engine supports `time_exit_bar` in position management
- [ ] Engine calls `update_calibration()` on stateful strategies
- [ ] All unit tests pass
- [ ] Integration test with all 54 strategies passes
- [ ] README updated with new strategies
