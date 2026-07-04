# Kairos Framework: Extended Strategy Implementation Specifications

**Version:** 1.1
**Date:** 2026-06-30
**Scope:** 50 additional strategies (Crypto, Forex, Stocks, Universal)

---

## Table of Contents

1. [Crypto-Specific Strategies (1-10)](#1-crypto-specific-strategies)
2. [Forex-Specific Strategies (11-20)](#2-forex-specific-strategies)
3. [Stock-Specific Strategies (21-32)](#3-stock-specific-strategies)
4. [Universal / Cross-Asset Strategies (33-50)](#4-universal--cross-asset-strategies)
5. [Implementation Order](#5-implementation-order)
6. [Engine Modifications](#6-engine-modifications)

---

## 1. Crypto-Specific Strategies

### 1.1 Funding Rate Arbitrage

**ID:** `funding_rate_arbitrage`
**Module:** `kairos_crypto.py` (new)
**Type:** Delta-neutral arbitrage

**Mechanism:**
Long spot + short perpetual futures when the funding rate is positive and predicted to persist. Collect funding payments every 8 hours while remaining delta-neutral. The edge comes from the funding rate being a predictable cost of carry that mean-reverts slowly.

**Kairos Integration:**
Use the predicted close distribution to estimate the directional risk of holding the position. If the predicted range is tight (low volatility) and the funding rate is positive, the probability of a large adverse move is low. Size the position inversely to predicted volatility.

**Algorithm:**
```
1. Get current funding rate from context["funding_rate"]
2. If funding_rate < min_funding_threshold: return None
3. pred_range = (dist.stats["close"]["pct_90"] - dist.stats["close"]["pct_10"]) / current_price
4. If pred_range > max_volatility_threshold: return None (directional risk too high)
5. If predicted close mean < current_price: return None (predicted down = short perp loses)
6. direction = FLAT (delta-neutral)
7. size = min(funding_rate / pred_range, max_position_size)
8. metadata: action="long_spot_short_perp", funding_rate, predicted_range
```

**Constructor:**
```python
def __init__(self, min_funding_threshold: float = 0.0001,
             max_volatility_threshold: float = 0.02,
             max_position_size: float = 0.3)
```

**Acceptance Criteria:**
- Only enters when funding rate > 0.01% and predicted range < 2%
- Returns FLAT signal with metadata for dual-leg execution
- Size capped at 30% of capital

### 1.2 Basis Trade

**ID:** `basis_trade`
**Module:** `kairos_crypto.py`
**Type:** Futures-spread arbitrage

**Mechanism:**
Short the futures premium when it is wide, long spot. Capture the convergence as the futures price approaches spot at expiry. The premium exists because futures buyers are paying for leverage.

**Kairos Integration:**
Compare the predicted close distribution to the futures mark price. If the predicted close mean is significantly below the futures mark, the premium is likely to compress. Enter the basis trade.

**Algorithm:**
```
1. futures_mark = context["futures_mark_price"]
2. if futures_mark is None: return None
3. basis = (futures_mark - current_price) / current_price
4. if basis < min_basis: return None
5. predicted_close = dist.stats["close"]["mean"]
6. if predicted_close > futures_mark * 0.995: return None (premium may widen)
7. confidence = basis / predicted_range
8. return FLAT signal with metadata: action="short_basis"
```

**Constructor:**
```python
def __init__(self, min_basis: float = 0.005,
             max_basis: float = 0.15)
```

### 1.3 Stablecoin Depeg

**ID:** `stablecoin_depeg`
**Module:** `kairos_crypto.py`
**Type:** Mean reversion arbitrage

**Mechanism:**
Stablecoins occasionally deviate from $1.00 due to liquidity crises or exchange-specific issues. Trade the mean reversion: buy below $0.995, sell above $1.005. The edge is that stablecoins are designed to return to peg.

**Kairos Integration:**
Kairos predicts the recovery path. If the predicted close distribution is centered inside the peg band ($0.995-$1.005), the depeg is temporary. Enter in the direction of the predicted recovery.

**Algorithm:**
```
1. if current_price < 0.995:
       direction = LONG
       target = 1.0
       stop = current_price * 0.99
   elif current_price > 1.005:
       direction = SHORT
       target = 1.0
       stop = current_price * 1.01
   else: return None
2. pred_mean = dist.stats["close"]["mean"]
3. if not (0.995 <= pred_mean <= 1.005): return None (predicted persistent depeg)
4. ev = dist.expected_value(current_price, target, stop)
5. if ev <= 0: return None
6. return Signal with high confidence (mean reversion is reliable)
```

**Constructor:**
```python
def __init__(self, lower_peg: float = 0.995,
             upper_peg: float = 1.005,
             max_deviation: float = 0.05)
```

### 1.4 Exchange Spread Arbitrage

**ID:** `exchange_spread`
**Module:** `kairos_crypto.py`
**Type:** Cross-exchange arbitrage

**Mechanism:**
Same asset trades at slightly different prices on different exchanges. Buy low, sell high simultaneously. Requires low latency, but the predicted distribution can identify persistent spreads.

**Kairos Integration:**
Run Kairos on both exchange price feeds. If the predicted low on exchange A is above the predicted high on exchange B, a risk-free arbitrage exists. The 60-sample distribution gives the probability of the spread persisting.

**Algorithm:**
```
1. Requires context["other_exchange_dist"] and context["other_exchange_price"]
2. pred_low_A = dist.stats["low"]["mean"]
3. pred_high_B = other_dist.stats["high"]["mean"]
4. if pred_low_A > pred_high_B * (1 + fee_buffer):
       return FLAT with metadata: action="arb_A_to_B"
5. if pred_high_B > pred_low_A * (1 + fee_buffer):
       return FLAT with metadata: action="arb_B_to_A"
6. return None
```

**Constructor:**
```python
def __init__(self, fee_buffer: float = 0.002)
```

### 1.5 Liquidation Cluster Front-Run

**ID:** `liquidation_front_run`
**Module:** `kairos_crypto.py`
**Type:** Microstructure exploitation

**Mechanism:**
Crypto markets have visible liquidation clusters at round-number leverage levels. When price approaches a cluster, stop-hunts trigger cascading liquidations. Front-run the wick and reversal.

**Kairos Integration:**
If the predicted high/low percentile lands exactly on a known liquidation wall, the model is predicting a wick to that level followed by a reversal. Place limit orders just inside the predicted extreme.

**Algorithm:**
```
1. liq_walls = context["liquidation_walls"]  # list of price levels
2. pred_high = dist.stats["high"]["pct_90"]
3. pred_low = dist.stats["low"]["pct_10"]
4. for wall in liq_walls:
       if abs(pred_high - wall) / current_price < proximity:
           return SHORT signal, target = wall, stop = wall * 1.02
       if abs(pred_low - wall) / current_price < proximity:
           return LONG signal, target = wall, stop = wall * 0.98
5. return None
```

**Constructor:**
```python
def __init__(self, proximity: float = 0.005)
```

### 1.6 Funding Rate Prediction

**ID:** `funding_rate_prediction`
**Module:** `kairos_crypto.py`
**Type:** Predictive funding arbitrage

**Mechanism:**
Funding rates are calculated from the premium index over the past 8 hours. Predict the next funding rate from the price path, then trade the perp before the market reprices.

**Kairos Integration:**
Kairos predicts the price path that determines the funding rate. If the predicted mean is above current price for the next 8 hours, the funding rate will likely be positive. Short the perp early to collect the funding.

**Algorithm:**
```
1. pred_mean = dist.stats["close"]["mean"]
2. if pred_mean > current_price * 1.002:
       predicted_funding = positive
       direction = SHORT (collect funding)
   elif pred_mean < current_price * 0.998:
       predicted_funding = negative
       direction = LONG (collect funding)
   else: return None
3. stop = dist.stats["low"]["pct_10"] if direction == LONG else dist.stats["high"]["pct_90"]
4. target = current_price * 1.01 if direction == LONG else current_price * 0.99
```

**Constructor:**
```python
def __init__(self, funding_threshold: float = 0.002)
```

### 1.7 On-Chain Exchange Flow Filter

**ID:** `onchain_flow_filter`
**Module:** `kairos_crypto.py`
**Type:** Filter wrapper

**Mechanism:**
Large exchange inflows indicate selling pressure (deposit to sell). Large outflows indicate accumulation (withdraw to hold). Use as a directional filter for Kairos signals.

**Kairos Integration:**
Wrap any base strategy. Only pass the signal if the on-chain flow aligns with the predicted direction. If predicted up but inflows are high, the signal is likely false.

**Algorithm:**
```
1. inflow = context["exchange_inflow"]
2. outflow = context["exchange_outflow"]
3. net_flow = inflow - outflow
4. base_signal = base_strategy.generate_signal(...)
5. if base_signal is None: return None
6. if base_signal.direction == LONG and net_flow > 0: return None (selling pressure)
7. if base_signal.direction == SHORT and net_flow < 0: return None (accumulation)
8. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy)
```

### 1.8 Options Gamma Squeeze

**ID:** `gamma_squeeze`
**Module:** `kairos_crypto.py`
**Type:** Options market microstructure

**Mechanism:**
If the predicted close is near a strike with heavy gamma exposure, market makers will hedge by buying/selling the underlying as price moves, amplifying the move. This is the gamma squeeze.

**Kairos Integration:**
Use the predicted close distribution to find which strike has the most gamma concentration. If the predicted mean is moving toward that strike, the gamma squeeze will accelerate the move. Trade in that direction.

**Algorithm:**
```
1. gamma_map = context["gamma_by_strike"]  # dict: strike -> gamma
2. if gamma_map is None: return None
3. pred_mean = dist.stats["close"]["mean"]
4. Find strike with max gamma near pred_mean
5. if pred_mean > current_price and strike > current_price:
       direction = LONG (gamma squeeze up)
   elif pred_mean < current_price and strike < current_price:
       direction = SHORT (gamma squeeze down)
   else: return None
6. size = max_gamma / total_gamma  # proportional to squeeze intensity
```

**Constructor:**
```python
def __init__(self, strike_proximity: float = 0.01)
```

### 1.9 Hash Rate Difficulty Filter

**ID:** `hash_rate_filter`
**Module:** `kairos_crypto.py`
**Type:** Regime filter (BTC-specific)

**Mechanism:**
BTC hash rate drops indicate miner capitulation, which historically marks local bottoms. Hash rate recovery indicates miner confidence. Use as a long-only regime filter.

**Kairos Integration:**
Only take long signals from Kairos when hash rate is recovering (7-day MA > 30-day MA). Skip all signals when hash rate is declining.

**Algorithm:**
```
1. hash_rate_ma7 = context["hash_rate_ma7"]
2. hash_rate_ma30 = context["hash_rate_ma30"]
3. if hash_rate_ma7 < hash_rate_ma30: return None (miner capitulation)
4. base_signal = base_strategy.generate_signal(...)
5. if base_signal and base_signal.direction == SHORT: return None
6. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy)
```

### 1.10 Perp-Spot Funding Harvest

**ID:** `funding_harvest`
**Module:** `kairos_crypto.py`
**Type:** Cross-asset carry

**Mechanism:**
Rotate between assets with the highest predicted funding yield. The 'return' is the funding rate, not price appreciation. Build a portfolio of delta-neutral positions that maximizes funding income.

**Kairos Integration:**
Cross-asset Sharpe ranking, but the expected return is the predicted funding rate divided by predicted volatility. Assets with high funding and low predicted volatility are optimal.

**Algorithm:**
```
1. For each asset in multi_asset_predictions:
       funding_yield = context["funding_rates"][asset]
       pred_vol = pred.dist.stats["close"]["std"] / pred.current_price
       carry_sharpe = funding_yield / pred_vol
2. Rank by carry_sharpe
3. Top N assets get FLAT signals with metadata: action="funding_harvest"
```

**Constructor:**
```python
def __init__(self, top_n: int = 3,
             min_carry_sharpe: float = 0.1)
```

---

## 2. Forex-Specific Strategies

### 2.1 Carry Trade

**ID:** `carry_trade`
**Module:** `kairos_forex.py` (new)
**Type:** Interest rate differential arbitrage

**Mechanism:**
Borrow a low-yield currency (JPY, CHF), lend a high-yield currency (USD, AUD, TRY). Hold overnight to accumulate swap points. The edge is the interest rate differential minus exchange rate risk.

**Kairos Integration:**
Predicted range tight + wide interest rate differential = high carry-to-risk ratio. Size the position inversely to predicted volatility. If predicted range > differential, the trade is negative EV.

**Algorithm:**
```
1. base_rate = context["base_interest_rate"]
2. quote_rate = context["quote_interest_rate"]
3. differential = base_rate - quote_rate
4. pred_range = (dist.stats["close"]["pct_90"] - dist.stats["close"]["pct_10"]) / current_price
5. carry_to_risk = differential / pred_range
6. if carry_to_risk < min_ratio: return None
7. direction = LONG if differential > 0 else SHORT
8. size = min(carry_to_risk * 0.1, max_size)
```

**Constructor:**
```python
def __init__(self, min_ratio: float = 0.5,
             max_size: float = 0.3)
```

### 2.2 Session Breakout

**ID:** `session_breakout`
**Module:** `kairos_forex.py`
**Type:** Time-based breakout

**Mechanism:**
Forex has three major sessions: Tokyo, London, New York. The overlap between London and NY (8am-12pm EST) is the most volatile. Predicted range expansion during session overlap = high-probability breakout.

**Kairos Integration:**
Use the predicted range width as a session-volatility predictor. If the predicted range is significantly wider than the Asian session range, enter in the direction of the predicted close before the overlap begins.

**Algorithm:**
```
1. asian_range = context["asian_session_range"]
2. pred_range = dist.stats["close"]["pct_90"] - dist.stats["close"]["pct_10"]
3. if pred_range < asian_range * 1.5: return None
4. direction = LONG if dist.stats["close"]["mean"] > current_price else SHORT
5. entry = current_price
6. stop = dist.stats["low"]["pct_10"] if LONG else dist.stats["high"]["pct_90"]
7. target = dist.stats["high"]["pct_90"] if LONG else dist.stats["low"]["pct_10"]
```

**Constructor:**
```python
def __init__(self, range_multiplier: float = 1.5)
```

### 2.3 London Fix Fade

**ID:** `london_fix_fade`
**Module:** `kairos_forex.py`
**Type:** Mean reversion at fix

**Mechanism:**
The 4pm London WM/Reuters fix creates artificial buying/selling pressure as institutions rebalance. Post-fix, the price often reverts.

**Kairos Integration:**
If the predicted close is far from the fix-time price, the model predicts reversion. Fade the fix move in the direction of the predicted close.

**Algorithm:**
```
1. fix_price = context["fix_price"]
2. if fix_price is None: return None
3. pred_mean = dist.stats["close"]["mean"]
4. if pred_mean < fix_price * 0.998:
       direction = SHORT (fade the fix up)
   elif pred_mean > fix_price * 1.002:
       direction = LONG (fade the fix down)
   else: return None
5. stop = fix_price * 1.01 if SHORT else fix_price * 0.99
6. target = pred_mean
```

**Constructor:**
```python
def __init__(self, fix_time: str = "16:00",
             fade_threshold: float = 0.002)
```

### 2.4 Central Bank Divergence

**ID:** `cb_divergence`
**Module:** `kairos_forex.py`
**Type:** Macro trend following

**Mechanism:**
Trade the spread between central bank policy rates. Wider divergence = stronger trend. The currency of the hawkish central bank appreciates against the dovish one.

**Kairos Integration:**
Use predicted Sharpe as a divergence-strength proxy. Higher predicted Sharpe = more conviction in the divergence trade. Combine with actual rate differentials for sizing.

**Algorithm:**
```
1. base_cb_rate = context["base_cb_rate"]
2. quote_cb_rate = context["quote_cb_rate"]
3. divergence = base_cb_rate - quote_cb_rate
4. pred_sharpe = dist.predicted_sharpe()
5. if divergence > 0 and pred_sharpe > 0.5:
       direction = LONG
   elif divergence < 0 and pred_sharpe > 0.5:
       direction = SHORT
   else: return None
6. size = min(abs(divergence) * pred_sharpe, max_size)
```

**Constructor:**
```python
def __init__(self, max_size: float = 0.4)
```

### 2.5 Safe Haven Rotation

**ID:** `safe_haven_rotation`
**Module:** `kairos_forex.py`
**Type:** Cross-asset regime rotation

**Mechanism:**
During risk-off periods, capital flows into safe havens (JPY, CHF, USD) and out of risk assets (AUD, NZD, EM currencies). Predicted close on safe havens vs risk assets determines rotation.

**Kairos Integration:**
Multi-asset: rank safe havens vs risk assets by predicted Sharpe. Rotate into safety when risk assets have negative predicted Sharpe. This is a cross-asset version of the existing CrossAssetRank.

**Algorithm:**
```
1. safe_havens = ["JPY", "CHF", "USD"]
2. risk_assets = ["AUD", "NZD", "TRY"]
3. safe_sharpe = mean(predicted_sharpe for safe havens)
4. risk_sharpe = mean(predicted_sharpe for risk assets)
5. if safe_sharpe > risk_sharpe + threshold:
       Long safe haven, short risk asset
   elif risk_sharpe > safe_sharpe + threshold:
       Long risk asset, short safe haven
   else: return None
```

**Constructor:**
```python
def __init__(self, threshold: float = 0.3)
```

### 2.6 Triangular Arbitrage

**ID:** `triangular_arbitrage`
**Module:** `kairos_forex.py`
**Type:** Cross-rate inefficiency

**Mechanism:**
Cross-rate inefficiencies: EUR/USD * USD/JPY should equal EUR/JPY. If the synthetic cross != actual, an arbitrage exists.

**Kairos Integration:**
Predicted close on all three legs. If the synthetic cross computed from predicted distributions deviates from the actual predicted cross, arb exists.

**Algorithm:**
```
1. Requires three Kairos distributions: EUR/USD, USD/JPY, EUR/JPY
2. synthetic = pred_eurusd * pred_usdjpy
3. actual = pred_eurjpy
4. deviation = abs(synthetic - actual) / actual
5. if deviation > threshold: return FLAT with metadata: action="tri_arb"
6. return None
```

**Constructor:**
```python
def __init__(self, threshold: float = 0.001)
```

### 2.7 Sovereign CDS Spread Filter

**ID:** `cds_spread_filter`
**Module:** `kairos_forex.py`
**Type:** Credit-led directional filter

**Mechanism:**
Widening sovereign CDS spreads indicate increased default risk, which weakens the currency. Use as a leading indicator for directional trades.

**Kairos Integration:**
Only short currencies with widening CDS and Kairos confirmation. Only long currencies with tightening CDS and Kairos confirmation.

**Algorithm:**
```
1. cds_change = context["cds_spread_change"]  # 5-day change
2. base_signal = base_strategy.generate_signal(...)
3. if base_signal is None: return None
4. if cds_change > 0 and base_signal.direction == LONG: return None
5. if cds_change < 0 and base_signal.direction == SHORT: return None
6. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy)
```

### 2.8 CFTC COT Positioning Filter

**ID:** `cot_positioning_filter`
**Module:** `kairos_forex.py`
**Type:** Contrarian positioning filter

**Mechanism:**
CFTC Commitment of Traders report shows commercials vs speculators positioning. Extreme speculator positioning is a contrarian signal.

**Kairos Integration:**
Fade extreme speculator positioning when Kairos predicts reversal. If speculators are extremely long and Kairos predicts down, take a short signal.

**Algorithm:**
```
1. spec_position = context["speculator_net_position"]  # normalized 0-1
2. base_signal = base_strategy.generate_signal(...)
3. if base_signal is None: return None
4. if spec_position > 0.8 and base_signal.direction == LONG: return None
5. if spec_position < 0.2 and base_signal.direction == SHORT: return None
6. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             extreme_threshold: float = 0.8)
```

### 2.9 Asian Range Breakout

**ID:** `asian_range_breakout`
**Module:** `kairos_forex.py`
**Type:** Session-based breakout

**Mechanism:**
The Tokyo session often establishes the daily range. If the predicted high/low is outside the Asian session range, a breakout is likely.

**Kairos Integration:**
Compare the predicted range to the Asian session high/low. If predicted high > Asian high or predicted low < Asian low, enter in the breakout direction.

**Algorithm:**
```
1. asian_high = context["asian_high"]
2. asian_low = context["asian_low"]
3. pred_high = dist.stats["high"]["mean"]
4. pred_low = dist.stats["low"]["mean"]
5. if pred_high > asian_high:
       direction = LONG, entry = asian_high, stop = asian_low
   elif pred_low < asian_low:
       direction = SHORT, entry = asian_low, stop = asian_high
   else: return None
6. target = pred_high if LONG else pred_low
```

**Constructor:**
```python
def __init__(self, confirmation_pct: float = 0.001)
```

### 2.10 Interest Rate Swap Spread

**ID:** `ois_swap_spread`
**Module:** `kairos_forex.py`
**Type:** Rates-forex cross-market

**Mechanism:**
The OIS ( Overnight Index Swap) curve reflects the market's expectation of central bank policy. A steepening curve = hawkish expectations = currency appreciation.

**Kairos Integration:**
Use Kairos on rate futures (e.g., SOFR futures, Euribor futures) to predict the spot FX direction. If predicted rates rise, the currency should appreciate.

**Algorithm:**
```
1. ois_curve = context["ois_curve"]  # list of rates by tenor
2. curve_slope = ois_curve[-1] - ois_curve[0]
3. pred_mean = dist.stats["close"]["mean"]
4. if curve_slope > 0 and pred_mean > current_price:
       direction = LONG
   elif curve_slope < 0 and pred_mean < current_price:
       direction = SHORT
   else: return None
5. size = min(abs(curve_slope) * 10, max_size)
```

**Constructor:**
```python
def __init__(self, max_size: float = 0.3)
```

---

## 3. Stock-Specific Strategies

### 3.1 Post-Earnings Announcement Drift (PEAD)

**ID:** `pead`
**Module:** `kairos_stocks.py` (new)
**Type:** Event-driven momentum

**Mechanism:**
Stocks drift in the direction of earnings surprise for approximately 60 days post-announcement. The drift is strongest for extreme surprises (SUE > 2 or SUE < -2).

**Kairos Integration:**
Predicted close post-earnings > pre-earnings = long. Use the predicted range for position sizing. If the predicted range is tight, the drift is likely to be smooth and predictable. If wide, the drift may be noisy.

**Algorithm:**
```
1. sue = context["standardized_unexpected_earnings"]
2. if abs(sue) < 1.5: return None (not extreme enough)
3. pred_mean = dist.stats["close"]["mean"]
4. if sue > 0 and pred_mean > current_price:
       direction = LONG
   elif sue < 0 and pred_mean < current_price:
       direction = SHORT
   else: return None
5. hold_days = 20  # typical drift window
6. size = min(abs(sue) * 0.15, max_size)
```

**Constructor:**
```python
def __init__(self, min_sue: float = 1.5,
             max_size: float = 0.3,
             hold_days: int = 20)
```

### 3.2 Earnings Momentum (SUE + EAR)

**ID:** `earnings_momentum`
**Module:** `kairos_stocks.py`
**Type:** Composite earnings signal

**Mechanism:**
Combine Standardized Unexpected Earnings (SUE) with Earnings Announcement Return (EAR). High SUE + High EAR = strongest momentum. Rank stocks by composite score.

**Kairos Integration:**
Kairos predicts the post-announcement path. Enter on day 2 if the prediction confirms the earnings surprise direction. The predicted Sharpe determines position size.

**Algorithm:**
```
1. sue = context["sue"]
2. ear = context["ear"]  # earnings announcement day return
3. composite = sue * 0.6 + ear * 0.4
4. if composite < threshold: return None
5. direction = LONG if composite > 0 else SHORT
6. pred_sharpe = dist.predicted_sharpe()
7. size = min(abs(composite) * pred_sharpe * 0.2, max_size)
```

**Constructor:**
```python
def __init__(self, threshold: float = 1.0,
             max_size: float = 0.3)
```

### 3.3 Index Rebalancing Arbitrage

**ID:** `index_rebalance`
**Module:** `kairos_stocks.py`
**Type:** Index inclusion/exclusion event

**Mechanism:**
Index additions create forced buying (index funds must buy). Index deletions create forced selling. Front-run the rebalancing.

**Kairos Integration:**
Predicted high on addition day = front-run index funds. Predicted low on deletion day = front-run the sell pressure. Use predicted close to estimate the magnitude of the rebalancing move.

**Algorithm:**
```
1. event = context["index_event"]  # "addition" or "deletion"
2. if event is None: return None
3. pred_mean = dist.stats["close"]["mean"]
4. if event == "addition" and pred_mean > current_price:
       direction = LONG
   elif event == "deletion" and pred_mean < current_price:
       direction = SHORT
   else: return None
5. size = 0.5  # high conviction event-driven
```

**Constructor:**
```python
def __init__(self, event_window_days: int = 3)
```

### 3.4 Sector Rotation Momentum

**ID:** `sector_rotation`
**Module:** `kairos_stocks.py`
**Type:** Cross-sector allocation

**Mechanism:**
Rotate into sectors with the highest predicted Sharpe, out of sectors with the lowest. This is a natural extension of cross-asset ranking to sector ETFs.

**Kairos Integration:**
Run Kairos on sector ETFs (XLK, XLF, XLE, etc.). Rank by predicted Sharpe. Long the top 2, short the bottom 2. Rebalance weekly.

**Algorithm:**
```
1. sector_predictions = context["sector_predictions"]  # dict of AssetPrediction
2. rankings = [(symbol, pred.dist.predicted_sharpe()) for symbol, pred in sector_predictions.items()]
3. rankings.sort(key=lambda x: x[1], reverse=True)
4. long_sectors = rankings[:2]
5. short_sectors = rankings[-2:]
6. Return signals for each sector
```

**Constructor:**
```python
def __init__(self, top_n: int = 2,
             bottom_n: int = 2)
```

### 3.5 Pairs Trading (Cointegration)

**ID:** `cointegration_pairs`
**Module:** `kairos_stocks.py`
**Type:** Statistical arbitrage

**Mechanism:**
Two cointegrated stocks diverge temporarily. Trade the convergence: short the outperforming stock, long the underperforming stock. The hedge ratio is derived from the cointegration regression.

**Kairos Integration:**
Use Kairos on the spread instead of price. Predicted spread mean reversion = signal. The predicted distribution of the spread gives the stop and target levels directly.

**Algorithm:**
```
1. spread = stock_A - hedge_ratio * stock_B
2. spread_dist = KairosDistribution(predictions_on_spread)
3. pred_mean = spread_dist.stats["close"]["mean"]
4. if pred_mean > current_spread:
       direction = SHORT spread (short A, long B)
   elif pred_mean < current_spread:
       direction = LONG spread (long A, short B)
   else: return None
5. stop = spread_dist.stats["high"]["pct_90"] if SHORT else spread_dist.stats["low"]["pct_10"]
6. target = spread_dist.stats["low"]["pct_10"] if SHORT else spread_dist.stats["high"]["pct_90"]
```

**Constructor:**
```python
def __init__(self, hedge_ratio: float,
             pair_symbol: str)
```

### 3.6 Merger Arbitrage

**ID:** `merger_arb`
**Module:** `kairos_stocks.py`
**Type:** Event-driven arbitrage

**Mechanism:**
Long the target company, short the acquirer. Capture the deal spread. The risk is deal failure.

**Kairos Integration:**
Predicted close on target > current price = deal likely to close. If predicted close is near the offer price, the spread is safe. If predicted close is far below, deal risk is high.

**Algorithm:**
```
1. offer_price = context["offer_price"]
2. deal_prob = context["deal_probability"]
3. pred_mean = dist.stats["close"]["mean"]
4. if pred_mean >= offer_price * 0.98 and deal_prob > 0.8:
       direction = LONG target
       target = offer_price
       stop = current_price * 0.95
5. size = deal_prob * 0.5
```

**Constructor:**
```python
def __init__(self, min_deal_prob: float = 0.8)
```

### 3.7 Buyback Yield Capture

**ID:** `buyback_yield`
**Module:** `kairos_stocks.py`
**Type:** Price-support exploitation

**Mechanism:**
Companies with active buyback programs have price support at the buyback price floor. The company is a persistent buyer of its own stock.

**Kairos Integration:**
Use predicted low as a buyback-supported floor. If the predicted low is near the buyback floor and the predicted close is above, the floor is likely to hold. Long near the predicted low.

**Algorithm:**
```
1. buyback_floor = context["buyback_floor"]
2. pred_low = dist.stats["low"]["mean"]
3. pred_close = dist.stats["close"]["mean"]
4. if abs(pred_low - buyback_floor) / buyback_floor < 0.02 and pred_close > buyback_floor:
       direction = LONG
       entry = pred_low
       stop = buyback_floor * 0.98
       target = pred_close
5. else: return None
```

**Constructor:**
```python
def __init__(self, proximity: float = 0.02)
```

### 3.8 Short Interest Squeeze

**ID:** `short_squeeze`
**Module:** `kairos_stocks.py`
**Type:** Convexity exploitation

**Mechanism:**
High short interest + predicted up move = squeeze potential. Shorts are forced to cover, amplifying the move. High convexity, high reward.

**Kairos Integration:**
Filter: only take long signals if short interest > threshold and predicted Sharpe > 1.0. Size aggressively because the payoff is convex.

**Algorithm:**
```
1. short_interest = context["short_interest_ratio"]
2. if short_interest < 0.15: return None
3. base_signal = base_strategy.generate_signal(...)
4. if base_signal is None or base_signal.direction != LONG: return None
5. pred_sharpe = dist.predicted_sharpe()
6. if pred_sharpe < 1.0: return None
7. base_signal.size = min(base_signal.size * 1.5, max_size)
8. base_signal.metadata["squeeze_potential"] = short_interest * pred_sharpe
9. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             min_short_interest: float = 0.15,
             min_sharpe: float = 1.0,
             max_size: float = 0.6)
```

### 3.9 Insider Transaction Clustering

**ID:** `insider_cluster`
**Module:** `kairos_stocks.py`
**Type:** Informational edge filter

**Mechanism:**
Clustered insider buying (multiple insiders buying within 30 days) predicts future appreciation. Insiders have non-public information.

**Kairos Integration:**
Only take long signals when insider buying cluster is detected and Kairos confirms the direction. If insiders are selling, block all long signals.

**Algorithm:**
```
1. insider_signal = context["insider_signal"]  # +1 buying cluster, -1 selling cluster, 0 neutral
2. base_signal = base_strategy.generate_signal(...)
3. if base_signal is None: return None
4. if insider_signal == -1 and base_signal.direction == LONG: return None
5. if insider_signal == 1 and base_signal.direction == SHORT: return None
6. if insider_signal == 1: base_signal.size *= 1.2
7. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy)
```

### 3.10 Dark Pool Print Analysis

**ID:** `dark_pool_filter`
**Module:** `kairos_stocks.py`
**Type:** Institutional flow filter

**Mechanism:**
Large block trades in dark pools reveal institutional direction. Dark pool buying sentiment is a leading indicator.

**Kairos Integration:**
Use dark pool sentiment as a filter for Kairos signals. If dark pool sentiment is bullish and Kairos predicts up, confirm the signal. If bearish, block longs.

**Algorithm:**
```
1. dark_pool_sentiment = context["dark_pool_sentiment"]  # -1 to 1
2. base_signal = base_strategy.generate_signal(...)
3. if base_signal is None: return None
4. if dark_pool_sentiment < -0.3 and base_signal.direction == LONG: return None
5. if dark_pool_sentiment > 0.3 and base_signal.direction == SHORT: return None
6. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             sentiment_threshold: float = 0.3)
```

### 3.11 Share Buyback Announcement Drift

**ID:** `buyback_drift`
**Module:** `kairos_stocks.py`
**Type:** Post-announcement drift

**Mechanism:**
Similar to PEAD, but for buyback announcements. Stocks drift up after buyback announcements due to reduced float and price support.

**Kairos Integration:**
Kairos predicts the drift magnitude. Size by predicted range. If predicted range is tight, the drift is smooth and predictable.

**Algorithm:**
```
1. announcement_date = context["buyback_announcement_date"]
2. if not within 5 days of announcement: return None
3. pred_mean = dist.stats["close"]["mean"]
4. if pred_mean > current_price:
       direction = LONG
       target = pred_mean
       stop = current_price * 0.97
5. size = min(0.3, max_size)
```

**Constructor:**
```python
def __init__(self, max_size: float = 0.3,
             drift_window: int = 5)
```

### 3.12 Dividend Capture

**ID:** `dividend_capture`
**Module:** `kairos_stocks.py`
**Type:** Income strategy

**Mechanism:**
Buy before ex-dividend date, sell after. The stock price drops by the dividend amount on ex-div, but often recovers partially. The edge is capturing the dividend minus the price drop.

**Kairos Integration:**
Predicted close > entry + dividend = profitable capture. If the predicted close is below the pre-div price, the capture is unprofitable.

**Algorithm:**
```
1. dividend = context["dividend_amount"]
2. ex_div_date = context["ex_div_date"]
3. days_to_ex = (ex_div_date - today).days
4. if days_to_ex < 1 or days_to_ex > 5: return None
5. pred_close = dist.stats["close"]["mean"]
6. if pred_close > current_price + dividend * 0.5:
       direction = LONG
       target = pred_close
       stop = current_price - dividend
7. else: return None
```

**Constructor:**
```python
def __init__(self, min_recovery: float = 0.5)
```

---

## 4. Universal / Cross-Asset Strategies

### 4.1 Kalman Filter Pairs Trading

**ID:** `kalman_pairs`
**Module:** `kairos_universal.py` (new)
**Type:** Dynamic statistical arbitrage

**Mechanism:**
Traditional pairs trading uses a fixed hedge ratio. Kalman filter adapts the hedge ratio dynamically based on recent price evolution. Trade when the spread deviates from the filtered mean by more than a threshold.

**Kairos Integration:**
Run the Kalman filter on the spread of two assets. Use Kairos to predict the spread's next-day distribution. If the predicted spread mean reverts toward the Kalman-filtered state, enter the convergence trade.

**Algorithm:**
```
1. Update Kalman filter with latest prices of asset A and B
2. Get filtered spread mean and variance
3. spread_dist = KairosDistribution(predictions_on_spread)
4. pred_spread_mean = spread_dist.stats["close"]["mean"]
5. z_score = (pred_spread_mean - kalman_mean) / kalman_std
6. if z_score > entry_z: direction = SHORT spread
7. if z_score < -entry_z: direction = LONG spread
8. stop = kalman_mean + 2*kalman_std (opposite direction)
9. target = kalman_mean
```

**Constructor:**
```python
def __init__(self, pair_symbol: str,
             entry_z: float = 2.0,
             exit_z: float = 0.5)
```

### 4.2 Hurst Exponent Regime Switching

**ID:** `hurst_regime_switch`
**Module:** `kairos_universal.py`
**Type:** Meta-filter

**Mechanism:**
Hurst exponent H > 0.5 indicates persistent/trending series. H < 0.5 indicates mean-reverting. H = 0.5 is random walk. Use H to switch between trend and mean-reversion strategies.

**Kairos Integration:**
Compute Hurst on the 60 predicted close samples. Use as a meta-filter in the decision tree: H > 0.55 -> trend strategies, H < 0.45 -> mean-reversion strategies, 0.45-0.55 -> uncertain/no trade.

**Algorithm:**
```
1. Compute Hurst exponent on the 60 predicted close samples using R/S analysis
2. if H > 0.55:
       regime = "trend"
       use TrendFollowingStrategy, MomentumContinuationStrategy
   elif H < 0.45:
       regime = "range"
       use RangeTradingStrategy, FadeExtremeStrategy
   else:
       return None
3. Generate signal from selected strategy
4. metadata["hurst"] = H
```

**Constructor:**
```python
def __init__(self, trend_threshold: float = 0.55,
             mean_reversion_threshold: float = 0.45)
```

### 4.3 Copula-Based Dependence Trading

**ID:** `copula_pairs`
**Module:** `kairos_universal.py`
**Type:** Non-linear dependence arbitrage

**Mechanism:**
Correlation measures linear dependence. Copulas capture non-linear and tail dependence. Fit a copula (Gaussian, t-copula, or Archimedean) to the joint distribution of two assets. Trade when the conditional probability of deviation is extreme.

**Kairos Integration:**
Fit copula to historical returns. Use Kairos predicted marginals (the 60 samples) to compute conditional probabilities. If P(A up | B down) is extreme given the copula, trade the conditional move.

**Algorithm:**
```
1. Fit copula to historical returns of asset A and B
2. Transform predicted samples to uniform marginals via empirical CDF
3. Compute conditional probability: P(A up | B down) from copula
4. If conditional probability > 0.8: LONG A, SHORT B
5. If conditional probability < 0.2: SHORT A, LONG B
6. Size proportional to |conditional_prob - 0.5|
```

**Constructor:**
```python
def __init__(self, pair_symbol: str,
             copula_type: str = "t",
             prob_threshold: float = 0.8)
```

### 4.4 Cointegration with Error Correction

**ID:** `cointegration_ect`
**Module:** `kairos_universal.py`
**Type:** Enhanced pairs trading

**Mechanism:**
Johansen test for cointegration identifies multiple cointegrating vectors. The error correction term (ECT) measures the speed of reversion to the long-run equilibrium. Faster ECT = stronger signal.

**Kairos Integration:**
Use Kairos on the error correction term. Predicted ECT reversion speed determines hold time. If predicted ECT reverts quickly (within 1-2 days), enter aggressively. If slowly, size down or skip.

**Algorithm:**
```
1. Compute ECT from Johansen cointegration
2. ect_dist = KairosDistribution(predictions_on_ECT)
3. pred_ect = ect_dist.stats["close"]["mean"]
4. if pred_ect > 0 and pred_ect < current_ect * 0.5:
       direction = SHORT ECT (reversion)
       hold_days = 1 if fast_reversion else 3
5. stop = ect_dist.stats["high"]["pct_90"] if SHORT else ect_dist.stats["low"]["pct_10"]
6. target = 0 (equilibrium)
```

**Constructor:**
```python
def __init__(self, pair_symbol: str,
             fast_reversion_threshold: float = 0.5)
```

### 4.5 Regime-Switching HMM

**ID:** `hmm_regime`
**Module:** `kairos_universal.py`
**Type:** Latent regime detection

**Mechanism:**
Hidden Markov Model detects latent regimes (bull, bear, sideways) from observable features. The transition matrix gives the probability of regime change. Trade the regime-appropriate strategy.

**Kairos Integration:**
Use predicted distribution features (entropy, skew, CV, range) as HMM observations. The HMM outputs the most likely current regime and the probability of switching. Only enter when regime probability > 0.7.

**Algorithm:**
```
1. Extract features: [entropy, skew, cv, range, direction]
2. Feed to HMM, get regime probabilities
3. regime = argmax(probabilities)
4. if max_prob < 0.7: return None
5. if regime == "bull": use TrendFollowingStrategy
6. if regime == "bear": use TrendFollowingStrategy (SHORT)
7. if regime == "sideways": use RangeTradingStrategy
8. metadata["regime"] = regime, metadata["regime_prob"] = max_prob
```

**Constructor:**
```python
def __init__(self, n_regimes: int = 3,
             min_regime_prob: float = 0.7)
```

### 4.6 Wavelet Decomposition Momentum

**ID:** `wavelet_momentum`
**Module:** `kairos_universal.py`
**Type:** Frequency-domain trading

**Mechanism:**
Decompose price into frequency components using wavelet transform. The low-frequency component is the trend, high-frequency is noise. Trade when the trend component shows momentum and the noise component is low.

**Kairos Integration:**
Predicted close distribution gives the 'signal' component. Apply wavelet decomposition to the 60 predicted samples to extract the dominant cycle. If the trend cycle is strengthening and noise is low, enter.

**Algorithm:**
```
1. Apply DWT (Discrete Wavelet Transform) to 60 predicted close samples
2. Extract approximation (trend) and detail (noise) coefficients
3. trend_strength = std(approximation) / std(detail)
4. if trend_strength < threshold: return None (too noisy)
5. if approximation[-1] > approximation[-5]: direction = LONG
6. if approximation[-1] < approximation[-5]: direction = SHORT
7. size = min(trend_strength * 0.2, max_size)
```

**Constructor:**
```python
def __init__(self, wavelet: str = "db4",
             threshold: float = 2.0,
             max_size: float = 0.3)
```

### 4.7 Detrended Fluctuation Analysis (DFA)

**ID:** `dfa_persistence`
**Module:** `kairos_universal.py`
**Type:** Persistence filter

**Mechanism:**
DFA measures long-range correlations. Alpha > 0.5 = persistent (trend-following works). Alpha < 0.5 = anti-persistent (mean-reversion works). Alpha = 0.5 = random walk (no edge).

**Kairos Integration:**
Apply DFA to the 60 predicted close samples. Use alpha as a strategy selector: alpha > 0.55 -> trend strategies, alpha < 0.45 -> mean-reversion strategies. Similar to Hurst but more robust to non-stationarity.

**Algorithm:**
```
1. Compute DFA alpha on 60 predicted close samples
2. if alpha > 0.55:
       regime = "trend"
       strategy = TrendFollowingStrategy()
   elif alpha < 0.45:
       regime = "mean_reversion"
       strategy = RangeTradingStrategy()
   else: return None
3. signal = strategy.generate_signal(dist, current_price, history, context)
4. signal.metadata["dfa_alpha"] = alpha
5. return signal
```

**Constructor:**
```python
def __init__(self, trend_threshold: float = 0.55,
             mr_threshold: float = 0.45)
```

### 4.8 Transfer Entropy Causality

**ID:** `transfer_entropy`
**Module:** `kairos_universal.py`
**Type:** Lead-lag detection

**Mechanism:**
Transfer entropy measures directed information flow between two time series. If BTC transfer entropy to ETH is high, BTC leads ETH. Use the leader's signal to front-run the follower.

**Kairos Integration:**
Compute transfer entropy from BTC predicted distribution to ETH realized moves. If BTC leads ETH, use BTC's Kairos signal to generate ETH signals with a time lag.

**Algorithm:**
```
1. te = context["transfer_entropy"]  # precomputed TE from leader to follower
2. if te < min_te: return None
3. leader_signal = context["leader_signal"]  # Kairos signal from leader asset
4. if leader_signal is None: return None
5. lag = context["optimal_lag"]  # bars
6. if leader_signal.direction == LONG:
       direction = LONG follower
   else: direction = SHORT follower
7. size = leader_signal.size * te / max_te
```

**Constructor:**
```python
def __init__(self, leader_symbol: str,
             min_te: float = 0.1,
             max_te: float = 1.0)
```

### 4.9 Graph Neural Network Sector Rotation

**ID:** `gnn_sector_rotation`
**Module:** `kairos_universal.py`
**Type:** ML-based allocation

**Mechanism:**
Assets are nodes, correlations are edges. A GNN learns to propagate information across the graph and predict which cluster will outperform. The node features are predicted Sharpe ratios from Kairos.

**Kairos Integration:**
Use predicted Sharpe as node features for each asset. The GNN predicts the next outperforming cluster. Long the predicted top cluster, short the predicted bottom cluster.

**Algorithm:**
```
1. Build graph: nodes = assets, edges = correlation > 0.7
2. Node features = [pred_sharpe, pred_mean, pred_std, entropy]
3. Run GNN forward pass
4. node_scores = GNN output (predicted next-period return)
5. top_nodes = argmax(node_scores, top_n)
6. bottom_nodes = argmin(node_scores, bottom_n)
7. Long top_nodes, short bottom_nodes
```

**Constructor:**
```python
def __init__(self, top_n: int = 2,
             bottom_n: int = 2,
             correlation_threshold: float = 0.7)
```

### 4.10 Reinforcement Learning Meta-Controller

**ID:** `rl_meta_controller`
**Module:** `kairos_universal.py`
**Type:** Strategy selection agent

**Mechanism:**
Train an RL agent (PPO or DQN) to select which strategy to run based on the current state. The state is the Kairos distribution features. The action is the strategy index. The reward is the realized PnL.

**Kairos Integration:**
State space = [entropy, skew, CV, range, recent_pnl, win_rate, drawdown]. Action space = all 42 strategy indices. The agent learns which strategy works in which distribution state.

**Algorithm:**
```
1. state = extract_features(dist, history, context)
2. action = agent.predict(state)  # strategy index
3. strategy = all_strategies[action]
4. signal = strategy.generate_signal(dist, current_price, history, context)
5. After trade: reward = pnl, store (state, action, reward) in replay buffer
6. Periodically train agent on replay buffer
```

**Constructor:**
```python
def __init__(self, all_strategies: List[Strategy],
             agent_type: str = "ppo",
             train_frequency: int = 100)
```

### 4.11 Fractal Dimension Trading

**ID:** `fractal_dimension`
**Module:** `kairos_universal.py`
**Type:** Noise filter

**Mechanism:**
High fractal dimension = high noise, low predictability. Low fractal dimension = clear trend, high predictability. Use as a filter to avoid trading on noisy days.

**Kairos Integration:**
Compute fractal dimension on the 60 predicted close samples using the box-counting method. If fractal dimension > threshold, return None (too noisy). If < threshold, proceed with base strategy.

**Algorithm:**
```
1. Compute box-counting dimension on 60 predicted close samples
2. if fractal_dim > threshold: return None
3. base_signal = base_strategy.generate_signal(...)
4. base_signal.metadata["fractal_dim"] = fractal_dim
5. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             threshold: float = 1.5)
```

### 4.12 Lempel-Ziv Complexity

**ID:** `lz_complexity`
**Module:** `kairos_universal.py`
**Type:** Randomness filter

**Mechanism:**
LZ complexity measures how compressible a sequence is. High complexity = random, unpredictable. Low complexity = patterned, predictable. Use as an alternative to entropy.

**Kairos Integration:**
Apply LZ complexity to the 60-sample predicted close sequence. If complexity is high (normalized > 0.8), the market is random and no trade. If low, proceed with base strategy.

**Algorithm:**
```
1. Convert 60 predicted close samples to binary sequence (up/down)
2. Compute LZ complexity
3. normalize = lz_complexity / theoretical_max
4. if normalize > threshold: return None
5. base_signal = base_strategy.generate_signal(...)
6. base_signal.metadata["lz_complexity"] = normalize
7. return base_signal
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             threshold: float = 0.8)
```

### 4.13 Recurrence Quantification Analysis (RQA)

**ID:** `rqa_determinism`
**Module:** `kairos_universal.py`
**Type:** Determinism filter

**Mechanism:**
RQA quantifies deterministic structure in time series. High determinism = the series is not random, trend-following may work. High entropy = random, avoid trading.

**Kairos Integration:**
RQA on predicted paths. High determinism -> use trend strategies. High entropy/laminarity -> use mean-reversion or skip.

**Algorithm:**
```
1. Build recurrence matrix from 60 predicted close samples
2. Compute determinism (DET) and laminarity (LAM)
3. if DET > 0.7 and LAM > 0.5:
       regime = "trend"
   elif DET < 0.3 and LAM < 0.3:
       regime = "random"
       return None
   else: regime = "uncertain"
4. Select strategy based on regime
```

**Constructor:**
```python
def __init__(self, det_threshold: float = 0.7,
             lam_threshold: float = 0.5)
```

### 4.14 Mutual Information Feature Selection

**ID:** `mutual_information_weight`
**Module:** `kairos_universal.py`
**Type:** Feature weighting

**Mechanism:**
Find which historical features (RSI, MACD, volume, etc.) have the highest mutual information with future returns. Weight strategies by the MI of their primary feature.

**Kairos Integration:**
Use MI to weight technical indicators in the Kairos context. If RSI has high MI with returns, weight RSIFilterStrategy more heavily. If MACD has low MI, deprioritize MACDFilterStrategy.

**Algorithm:**
```
1. Compute MI between each feature and future returns over lookback
2. mi_scores = {"rsi": 0.3, "macd": 0.15, "volume": 0.25, ...}
3. For each strategy, get its primary feature
4. Weight strategy signal by mi_scores[feature]
5. Select highest weighted signal
```

**Constructor:**
```python
def __init__(self, feature_map: Dict[str, str],
             lookback: int = 100)
```

### 4.15 Gaussian Process Regression

**ID:** `gaussian_process`
**Module:** `kairos_universal.py`
**Type:** Probabilistic regression

**Mechanism:**
GP regression provides a non-parametric predictive distribution with uncertainty. Better than simple KDE for small samples because it encodes smoothness assumptions via the kernel.

**Kairos Integration:**
Use GP on the 60 samples to get a smooth predictive distribution with credible intervals. The GP mean is the predicted close, the GP std is the uncertainty. Use for more accurate EV and Kelly calculations.

**Algorithm:**
```
1. Fit GP (RBF kernel) to 60 predicted close samples
2. Query GP at current_price to get predictive mean and variance
3. pred_mean = GP_mean
4. pred_std = sqrt(GP_variance)
5. Use pred_mean and pred_std for signal generation (same as base strategy but with GP uncertainty)
6. The GP uncertainty is typically tighter than empirical std for small samples
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             kernel: str = "rbf",
             noise_level: float = 0.01)
```

### 4.16 Bayesian Structural Time Series

**ID:** `bsts_decomposition`
**Module:** `kairos_universal.py`
**Type:** Trend decomposition

**Mechanism:**
BSTS decomposes price into trend, seasonal, and cycle components. The trend component is the signal. Trade when the trend is strong and the cycle is aligned.

**Kairos Integration:**
Use Kairos predicted close as the observation in BSTS. The BSTS updates the trend belief. If the predicted close confirms the BSTS trend, enter. If it contradicts, skip.

**Algorithm:**
```
1. Run BSTS on historical prices to extract trend and cycle
2. pred_mean = dist.stats["close"]["mean"]
3. bsts_trend = context["bsts_trend"]  # current trend estimate
4. if pred_mean > current_price and bsts_trend > 0:
       direction = LONG (both confirm)
   elif pred_mean < current_price and bsts_trend < 0:
       direction = SHORT (both confirm)
   else: return None (contradiction)
5. size = min(abs(bsts_trend) * pred_sharpe, max_size)
```

**Constructor:**
```python
def __init__(self, max_size: float = 0.3)
```

### 4.17 Particle Filter Tracking

**ID:** `particle_filter`
**Module:** `kairos_universal.py`
**Type:** State estimation

**Mechanism:**
Particle filter tracks the 'true' price state through noisy observations. The 60 predicted samples are treated as particles. Resample by likelihood to get a filtered state estimate.

**Kairos Integration:**
Use the 60 samples as particles. The particle filter resamples them based on how well each particle explains the recent price history. The filtered mean is a more robust prediction than the simple mean.

**Algorithm:**
```
1. Particles = 60 predicted close samples
2. Weights = likelihood of each particle given recent price history
3. Resample particles by weights
4. filtered_mean = weighted mean of resampled particles
5. filtered_std = weighted std of resampled particles
6. Use filtered_mean and filtered_std for signal generation
7. The filtered estimate is typically more accurate than raw mean
```

**Constructor:**
```python
def __init__(self, base_strategy: Strategy,
             n_particles: int = 60)
```

### 4.18 Spectral Clustering for Asset Selection

**ID:** `spectral_clustering`
**Module:** `kairos_universal.py`
**Type:** Cluster-based allocation

**Mechanism:**
Cluster assets by predicted correlation using spectral clustering. Trade the cluster with the highest predicted Sharpe. Assets within the same cluster move together.

**Kairos Integration:**
Use predicted close distributions to compute forward-looking correlations (not historical). Spectral clustering on the correlation matrix identifies natural groupings. Long the strongest cluster, short the weakest.

**Algorithm:**
```
1. Compute predicted correlation matrix from multi-asset distributions
2. Apply spectral clustering to get K clusters
3. For each cluster, compute average predicted Sharpe
4. best_cluster = argmax(average_sharpe)
5. worst_cluster = argmin(average_sharpe)
6. Long all assets in best_cluster, short all in worst_cluster
7. Size proportional to within-cluster predicted Sharpe
```

**Constructor:**
```python
def __init__(self, n_clusters: int = 3,
             correlation_lookback: int = 30)
```

---

## 5. Implementation Order

### Phase 1: Low-Hanging Fruit (No new dependencies)

| # | Strategy | File | Effort |
|---|----------|------|--------|
| 1 | Funding Rate Arbitrage | kairos_crypto.py | Low |
| 2 | Basis Trade | kairos_crypto.py | Low |
| 3 | Stablecoin Depeg | kairos_crypto.py | Low |
| 4 | Carry Trade | kairos_forex.py | Low |
| 5 | Session Breakout | kairos_forex.py | Low |
| 6 | PEAD | kairos_stocks.py | Low |
| 7 | Short Interest Squeeze | kairos_stocks.py | Low |
| 8 | Hurst Regime Switching | kairos_universal.py | Low |
| 9 | DFA Persistence | kairos_universal.py | Low |
| 10 | Fractal Dimension | kairos_universal.py | Low |

### Phase 2: Medium Complexity (Minor engine changes)

| # | Strategy | File | Effort |
|---|----------|------|--------|
| 11 | Liquidation Front-Run | kairos_crypto.py | Medium |
| 12 | Funding Rate Prediction | kairos_crypto.py | Medium |
| 13 | On-Chain Flow Filter | kairos_crypto.py | Medium |
| 14 | London Fix Fade | kairos_forex.py | Medium |
| 15 | Central Bank Divergence | kairos_forex.py | Medium |
| 16 | Asian Range Breakout | kairos_forex.py | Medium |
| 17 | Earnings Momentum | kairos_stocks.py | Medium |
| 18 | Sector Rotation | kairos_stocks.py | Medium |
| 19 | Pairs Trading (Cointegration) | kairos_stocks.py | Medium |
| 20 | Kalman Filter Pairs | kairos_universal.py | Medium |
| 21 | Cointegration ECT | kairos_universal.py | Medium |
| 22 | LZ Complexity | kairos_universal.py | Medium |
| 23 | Mutual Information Weight | kairos_universal.py | Medium |
| 24 | Particle Filter | kairos_universal.py | Medium |

### Phase 3: High Complexity (New algorithms)

| # | Strategy | File | Effort |
|---|----------|------|--------|
| 25 | Exchange Spread Arbitrage | kairos_crypto.py | Medium |
| 26 | Gamma Squeeze | kairos_crypto.py | High |
| 27 | Hash Rate Filter | kairos_crypto.py | Low |
| 28 | Perp-Spot Funding Harvest | kairos_crypto.py | Medium |
| 29 | Safe Haven Rotation | kairos_forex.py | Medium |
| 30 | Triangular Arbitrage | kairos_forex.py | Medium |
| 31 | Sovereign CDS Filter | kairos_forex.py | Medium |
| 32 | COT Positioning Filter | kairos_forex.py | Medium |
| 33 | OIS Swap Spread | kairos_forex.py | Medium |
| 34 | Index Rebalance | kairos_stocks.py | Medium |
| 35 | Merger Arbitrage | kairos_stocks.py | Medium |
| 36 | Buyback Yield | kairos_stocks.py | Medium |
| 37 | Insider Cluster | kairos_stocks.py | Medium |
| 38 | Dark Pool Filter | kairos_stocks.py | Medium |
| 39 | Buyback Drift | kairos_stocks.py | Medium |
| 40 | Dividend Capture | kairos_stocks.py | Medium |
| 41 | Copula Pairs | kairos_universal.py | High |
| 42 | HMM Regime | kairos_universal.py | High |
| 43 | Wavelet Momentum | kairos_universal.py | Medium |
| 44 | Transfer Entropy | kairos_universal.py | Medium |
| 45 | GNN Sector Rotation | kairos_universal.py | High |
| 46 | RL Meta-Controller | kairos_universal.py | High |
| 47 | RQA Determinism | kairos_universal.py | Medium |
| 48 | Gaussian Process | kairos_universal.py | Medium |
| 49 | BSTS Decomposition | kairos_universal.py | Medium |
| 50 | Spectral Clustering | kairos_universal.py | Medium |

---

## 6. Engine Modifications

### 6.1 New Context Fields Required

The orchestrator must pass the following in the `context` dict for the new strategies:

| Field | Type | Strategies Using |
|-------|------|------------------|
| `funding_rate` | float | 1.1, 1.6, 1.10 |
| `futures_mark_price` | float | 1.2 |
| `other_exchange_dist` | KairosDistribution | 1.4 |
| `other_exchange_price` | float | 1.4 |
| `liquidation_walls` | List[float] | 1.5 |
| `exchange_inflow` | float | 1.7 |
| `exchange_outflow` | float | 1.7 |
| `gamma_by_strike` | Dict[float, float] | 1.8 |
| `hash_rate_ma7` | float | 1.9 |
| `hash_rate_ma30` | float | 1.9 |
| `base_interest_rate` | float | 2.1, 2.4 |
| `quote_interest_rate` | float | 2.1, 2.4 |
| `asian_session_range` | float | 2.2, 2.9 |
| `asian_high` | float | 2.9 |
| `asian_low` | float | 2.9 |
| `fix_price` | float | 2.3 |
| `base_cb_rate` | float | 2.4 |
| `quote_cb_rate` | float | 2.4 |
| `cds_spread_change` | float | 2.7 |
| `speculator_net_position` | float | 2.8 |
| `ois_curve` | List[float] | 2.10 |
| `standardized_unexpected_earnings` | float | 3.1 |
| `sue` | float | 3.2 |
| `ear` | float | 3.2 |
| `index_event` | str | 3.3 |
| `sector_predictions` | Dict[str, AssetPrediction] | 3.4 |
| `offer_price` | float | 3.6 |
| `deal_probability` | float | 3.6 |
| `buyback_floor` | float | 3.7 |
| `short_interest_ratio` | float | 3.8 |
| `insider_signal` | int | 3.9 |
| `dark_pool_sentiment` | float | 3.10 |
| `buyback_announcement_date` | pd.Timestamp | 3.11 |
| `dividend_amount` | float | 3.12 |
| `ex_div_date` | pd.Timestamp | 3.12 |
| `transfer_entropy` | float | 4.8 |
| `optimal_lag` | int | 4.8 |
| `leader_signal` | Signal | 4.8 |
| `bsts_trend` | float | 4.16 |

### 6.2 New Module Files

| File | Strategies | Description |
|------|-----------|-------------|
| `kairos_crypto.py` | 1.1-1.10 | Crypto-specific strategies |
| `kairos_forex.py` | 2.1-2.10 | Forex-specific strategies |
| `kairos_stocks.py` | 3.1-3.12 | Stock-specific strategies |
| `kairos_universal.py` | 4.1-4.18 | Cross-asset universal strategies |

### 6.3 StrategyRegistry Update

Add the new modules to `StrategyRegistry.build_all()`:

```python
from kairos_crypto import (
    FundingRateArbitrage, BasisTrade, StablecoinDepeg,
    ExchangeSpreadArbitrage, LiquidationFrontRun,
    FundingRatePrediction, OnChainFlowFilter,
    GammaSqueeze, HashRateFilter, FundingHarvest

from kairos_forex import (
    CarryTrade, SessionBreakout, LondonFixFade,
    CBDivergence, SafeHavenRotation, TriangularArbitrage,
    CDSSpreadFilter, COTPositioningFilter,
    AsianRangeBreakout, OISSwapSpread

from kairos_stocks import (
    PEAD, EarningsMomentum, IndexRebalance,
    SectorRotation, CointegrationPairs, MergerArbitrage,
    BuybackYield, ShortSqueeze, InsiderCluster,
    DarkPoolFilter, BuybackDrift, DividendCapture

from kairos_universal import (
    KalmanPairs, HurstRegimeSwitch, CopulaPairs,
    CointegrationECT, HMMRegime, WaveletMomentum,
    DFAPersistence, TransferEntropy, GNNSectorRotation,
    RLMetaController, FractalDimension, LZComplexity,
    RQADeterminism, MutualInformationWeight,
    GaussianProcess, BSTSDecomposition, ParticleFilter,
    SpectralClustering
```

### 6.4 Testing Plan

For each new strategy, create a synthetic test:

```python
def test_<strategy_name>():
    # Create synthetic distribution and context
    # Verify signal generation matches expected logic
    # Verify None returned when conditions not met
    # Verify metadata contains required fields
```

Integration test: Run all 92 strategies (42 existing + 50 new) through the orchestrator with synthetic data. Verify no exceptions and all expected result keys present.

---

## 7. Summary

This document specifies 50 additional strategies for the Kairos framework, organized by asset class:

- **Crypto (10):** Funding rate, basis, depeg, exchange spread, liquidation front-run, funding prediction, on-chain flow, gamma squeeze, hash rate filter, funding harvest
- **Forex (10):** Carry trade, session breakout, London fix fade, CB divergence, safe haven rotation, triangular arb, CDS filter, COT filter, Asian range breakout, OIS swap spread
- **Stocks (12):** PEAD, earnings momentum, index rebalance, sector rotation, cointegration pairs, merger arb, buyback yield, short squeeze, insider cluster, dark pool filter, buyback drift, dividend capture
- **Universal (18):** Kalman pairs, Hurst switching, copula pairs, cointegration ECT, HMM regime, wavelet momentum, DFA, transfer entropy, GNN rotation, RL meta-controller, fractal dimension, LZ complexity, RQA, mutual information, Gaussian process, BSTS, particle filter, spectral clustering

**Total framework size after implementation: 92 strategies across 10 modules.**
