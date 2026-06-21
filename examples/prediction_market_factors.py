"""
prediction_market_factors.py

Comprehensive Kronos prediction with multi-dimensional market factor analysis,
using price_cache for all data fetching.

This is the price_cache counterpart to examples/akshare/prediction_akshare_2024-2025.py.
Data is pulled from price_cache's provider chain (no akshare import needed here);
the EnhancedMarketFactorAnalyzer layer is provider-agnostic and unchanged.

Usage:
    python prediction_market_factors.py

Modify STOCK_CONFIG at the bottom to target a different symbol.
"""

import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.append("../")
import price_cache
from kairos.data import get_forecast_window

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("WARNING: Cannot import Kronos model; prediction functionality unavailable")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams["axes.unicode_minus"] = False


# ==================== Market Factor Analyzer ====================

class EnhancedMarketFactorAnalyzer:
    """Multi-dimensional market factor analyzer."""

    def __init__(self):
        self.market_data = {}

    def analyze_market_trend(self, index_codes=("000001", "399001")):
        """Broad market trend via price_cache index data."""
        try:
            print("Comprehensive broad market trend analysis ...")
            market_analysis = {}

            for index_code in index_codes:
                index_name = "SSE Index" if index_code == "000001" else "SZSE Component Index"
                try:
                    raw = price_cache.get_price_data(index_code, interval="1d")
                    if raw is None or raw.empty:
                        continue

                    raw = raw.rename(columns=str.lower)
                    raw = raw.sort_index()

                    raw["ma5"] = raw["close"].rolling(5).mean()
                    raw["ma20"] = raw["close"].rolling(20).mean()
                    raw["ma60"] = raw["close"].rolling(60).mean()
                    raw["vol_ma5"] = raw["volume"].rolling(5).mean()

                    cur = raw.iloc[-1]
                    prev = raw.iloc[-2]

                    ma_bull = cur["ma5"] > cur["ma20"] > cur["ma60"]
                    above_ma20 = cur["close"] > cur["ma20"]
                    vol_ok = cur["volume"] > cur["vol_ma5"] * 0.8
                    strength = self._calc_trend_strength(raw)
                    uptrend = ma_bull and above_ma20 and strength > 0.6

                    market_analysis[index_name] = {
                        "is_main_uptrend": uptrend,
                        "trend_strength": strength,
                        "current_close": cur["close"],
                        "price_change_pct": (cur["close"] - prev["close"]) / prev["close"] * 100,
                        "market_status": "main_uptrend" if uptrend else "consolidation",
                    }
                except Exception:
                    pass

            if market_analysis:
                avg_strength = np.mean([d["trend_strength"] for d in market_analysis.values()])
                up_count = sum(1 for d in market_analysis.values() if d["is_main_uptrend"])
                overall_up = up_count >= len(market_analysis) * 0.5
                result = {
                    "overall_is_main_uptrend": overall_up,
                    "overall_trend_strength": avg_strength,
                    "detailed_analysis": market_analysis,
                    "market_status": "main_uptrend" if overall_up else "consolidation",
                }
                print(f"Market analysis: {result['market_status']}, "
                      f"trend strength: {avg_strength:.2f}")
                return result

        except Exception as e:
            print(f"Market analysis error: {e}")

        return self._default_market()

    def analyze_sector_resonance(self, stock_code):
        """Static sector resonance — extends easily with live data."""
        print("Analyzing sector resonance ...")

        hot_sectors = {
            "Robotics": {"momentum": 0.85, "active": True},
            "Semiconductors": {"momentum": 0.80, "active": True},
            "AI": {"momentum": 0.75, "active": True},
            "Low-altitude Economy": {"momentum": 0.70, "active": True},
            "New Energy": {"momentum": 0.60, "active": True},
            "Pharma": {"momentum": 0.50, "active": False},
        }

        # Very simple code→sector heuristic; replace with a live lookup as needed
        code_sectors = {
            "600580": ["Robotics", "Low-altitude Economy"],
            "300207": ["New Energy"],
            "300418": ["AI"],
            "000001": [],
            "600036": [],
        }
        matched = [
            {"sector": s, "momentum": hot_sectors[s]["momentum"], "is_active": hot_sectors[s]["active"]}
            for s in code_sectors.get(stock_code, [])
            if s in hot_sectors
        ]

        if matched:
            score = float(np.mean([m["momentum"] for m in matched]))
            hot = any(m["is_active"] for m in matched)
            main = max(matched, key=lambda x: x["momentum"])
        else:
            score, hot = 0.5, False
            main = {"sector": "Traditional Industry", "momentum": 0.5}

        result = {
            "matched_sectors": matched,
            "main_sector": main,
            "is_sector_hot": hot,
            "resonance_score": score,
            "sector_count": len(matched),
        }
        print(f"Sector analysis: {len(matched)} hot sectors matched, score: {score:.2f}")
        return result

    def analyze_macro_factors(self):
        """Static macro factor snapshot — update as conditions change."""
        print("Analyzing macro factors ...")
        us_rate = {
            "trend": "rate_cut_cycle",
            "expected_cuts_2025": 2,
        }
        macro = {
            "us_rate_cycle": us_rate,
            "domestic_policy": {"monetary_policy": "accommodative"},
            "overall_macro_score": 0.75,
        }
        print(f"Macro: {us_rate['trend']}, score: {macro['overall_macro_score']:.2f}")
        return macro

    def analyze_company_fundamentals(self, stock_code):
        """Per-stock fundamental scores — extend as needed."""
        fundamentals_db = {
            "600580": {"company_name": "Wolong Electric Drive",
                       "investment_rating": "positive_attention",
                       "fundamental_score": 0.70},
        }
        result = fundamentals_db.get(stock_code, {
            "company_name": "Unknown", "investment_rating": "neutral",
            "fundamental_score": 0.50,
        })
        print(f"Fundamentals: {result['company_name']}, score: {result['fundamental_score']:.2f}")
        return result

    # --- helpers ---

    def _calc_trend_strength(self, df):
        if len(df) < 20:
            return 0.5
        ma_slope = (df["ma5"].iloc[-1] - df["ma5"].iloc[-20]) / df["ma5"].iloc[-20]
        price_slope = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]
        vol_trend = df["volume"].iloc[-5:].mean() / df["volume"].iloc[-10:-5].mean()
        strength = ma_slope * 0.4 + price_slope * 0.4 + min(vol_trend - 1, 0.2) * 0.2
        return float(max(0.0, min(1.0, strength * 10)))

    def _default_market(self):
        return {"overall_is_main_uptrend": False, "overall_trend_strength": 0.5,
                "market_status": "unknown", "detailed_analysis": {}}


# ==================== Adjustment Factor ====================

def calculate_adjustment_factor(market, sector, macro, fundamental):
    base = 1.0

    trend = market["overall_trend_strength"]
    base *= 1 + (trend * 0.08 if market["overall_is_main_uptrend"] else (trend - 0.5) * 0.04)

    score = sector["resonance_score"]
    n = sector["sector_count"]
    if sector["is_sector_hot"]:
        base *= 1 + score * 0.06 + min(n * 0.01, 0.03)
    else:
        base *= 1 + (score - 0.5) * 0.02

    macro_score = macro["overall_macro_score"]
    base *= 1 + (macro_score - 0.5) * 0.06

    cuts = macro["us_rate_cycle"].get("expected_cuts_2025", 0)
    if macro["us_rate_cycle"]["trend"] == "rate_cut_cycle":
        base *= 1 + cuts * 0.015

    fund_score = fundamental["fundamental_score"]
    base *= 1 + (fund_score - 0.5) * 0.08

    return float(max(0.85, min(1.15, base)))


def enhance_prediction(pred_df, stock_code, analyzer):
    print("\nEnhancing prediction with market factors ...")
    market = analyzer.analyze_market_trend()
    sector = analyzer.analyze_sector_resonance(stock_code)
    macro = analyzer.analyze_macro_factors()
    fund = analyzer.analyze_company_fundamentals(stock_code)

    factor = calculate_adjustment_factor(market, sector, macro, fund)
    print(f"Composite adjustment factor: {factor:.4f}")

    enhanced = pred_df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in enhanced.columns:
            raw_adj = enhanced[col] * factor
            ratio = raw_adj / enhanced[col]
            if ratio.max() > 1.1:
                raw_adj = enhanced[col] * 1.1
            elif ratio.min() < 0.9:
                raw_adj = enhanced[col] * 0.9
            enhanced[col] = raw_adj
    if "volume" in enhanced.columns:
        enhanced["volume"] *= 1 + (factor - 1) * 0.3

    info = {
        "market_analysis": market,
        "sector_analysis": sector,
        "macro_analysis": macro,
        "fundamental_analysis": fund,
        "adjustment_factor": factor,
    }
    return enhanced, info


# ==================== Visualisation ====================

def plot_comprehensive(x_df, x_timestamp, pred_df, enhanced_df,
                       y_timestamp, stock_code, stock_name, enhancement_info,
                       output_dir):
    os.makedirs(output_dir, exist_ok=True)

    hist_close = pd.Series(x_df["close"].values, index=x_timestamp)
    base_close = pd.Series(pred_df["close"].values, index=y_timestamp)
    enh_close = pd.Series(enhanced_df["close"].values, index=y_timestamp)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    ax1.plot(hist_close.index[-200:], hist_close.values[-200:],
             color="#1f77b4", linewidth=2, label="Historical")
    ax1.plot(base_close.index, base_close.values,
             color="#ff7f0e", linewidth=2, label="Base Prediction")
    ax1.plot(enh_close.index, enh_close.values,
             color="#2ca02c", linewidth=2, linestyle="--", label="Enhanced Prediction")
    ax1.axvline(x=y_timestamp.iloc[0], color="red", linestyle="--", alpha=0.6)
    factor = enhancement_info["adjustment_factor"]
    ax1.set_title(f"{stock_name}({stock_code}) — Comprehensive Factor Prediction  "
                  f"(factor={factor:.3f})", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Close Price (CNY)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    scores = [
        enhancement_info["market_analysis"]["overall_trend_strength"],
        enhancement_info["sector_analysis"]["resonance_score"],
        enhancement_info["macro_analysis"]["overall_macro_score"],
        0.7 if enhancement_info["macro_analysis"]["us_rate_cycle"]["trend"] == "rate_cut_cycle" else 0.3,
        enhancement_info["fundamental_analysis"]["fundamental_score"],
    ]
    labels = ["Market", "Sector", "Macro", "US Rates", "Fundamentals"]
    ax2.bar(labels, scores, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#f39c12", "#9b59b6"],
            alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Score")
    ax2.set_title("Factor Scores")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(scores):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"{stock_code}_comprehensive_prediction.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {chart_path}")
    plt.close('all')


# ==================== Main ====================

def run_comprehensive_prediction(stock_code, stock_name, pred_days, output_dir,
                                  lookback=300, device="cpu"):
    print(f"\nStarting {stock_name}({stock_code}) comprehensive prediction")
    print("=" * 60)

    price_cache.configure(remote=False)
    analyzer = EnhancedMarketFactorAnalyzer()

    # 1. Fetch data
    print("Step 1: Fetching data via price_cache ...")
    x_df, x_timestamp, y_timestamp = get_forecast_window(
        symbol=stock_code,
        interval="1d",
        lookback=lookback,
        pred_len=pred_days,
        amount="auto",
    )
    print(f"Loaded {len(x_df)} bars "
          f"({x_timestamp.iloc[0].date()} → {x_timestamp.iloc[-1].date()})")

    # 2. Load model
    print("Step 2: Loading Kronos model ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # 3. Base prediction
    print("Step 3: Running base prediction ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_days,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    # 4. Market-factor enhancement
    print("Step 4: Applying market factor enhancement ...")
    enhanced_df, enhancement_info = enhance_prediction(pred_df, stock_code, analyzer)

    # 5. Visualise
    print("Step 5: Generating charts ...")
    plot_comprehensive(x_df, x_timestamp, pred_df, enhanced_df, y_timestamp,
                       stock_code, stock_name, enhancement_info, output_dir)

    # 6. Save report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock_code": stock_code,
        "adjustment_factor": enhancement_info["adjustment_factor"],
        "market_status": enhancement_info["market_analysis"]["market_status"],
    }
    report_path = os.path.join(output_dir, f"{stock_code}_analysis_report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    # 7. Summary
    current = float(x_df["close"].iloc[-1])
    base_end = float(pred_df["close"].iloc[-1])
    enh_end = float(enhanced_df["close"].iloc[-1])
    print(f"\nCurrent price:     {current:.2f}")
    print(f"Base prediction:   {base_end:.2f}  ({(base_end/current-1)*100:+.2f}%)")
    print(f"Enhanced predict:  {enh_end:.2f}  ({(enh_end/current-1)*100:+.2f}%)")
    print(f"Factor:            {enhancement_info['adjustment_factor']:.4f}")
    print(f"\nDone: {stock_name}({stock_code})")


if __name__ == "__main__":
    # Symbol format depends on your price_cache provider:
    #   yfinance (default): Shanghai stocks end in .SS, Shenzhen in .SZ
    #   akshare provider:   bare 6-digit codes (603288, 600580, etc.)
    STOCK_CONFIG = {
        "stock_code": "603288.SS",   # Haitian Flavouring — yfinance format
        "stock_name": "Haitian Flavouring",
        "pred_days": 60,
        "output_dir": "./output",
        "lookback": 300,
        "device": "cpu",
    }

    run_comprehensive_prediction(**STOCK_CONFIG)
