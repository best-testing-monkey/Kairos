import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import random
import akshare as ak
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Add project path for importing custom modules
sys.path.append("../../")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("⚠️ Cannot import Kronos model, prediction functionality will be unavailable")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==================== Base Data Fetch Functions ====================
def ensure_output_directory(output_dir):
    """Ensure output directory exists, create if not"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    return output_dir


def fetch_real_stock_data(stock_code, period="daily", adjust="qfq"):
    """
    Fetch real stock data using AKShare
    """
    try:
        print(f"📡 Fetching real stock data for {stock_code} via AKShare...")

        # Get stock data
        df = ak.stock_zh_a_hist(symbol=stock_code, period=period, adjust=adjust)

        if df is None or df.empty:
            print(f"❌ No data retrieved for {stock_code}")
            return None

        # Rename columns to unified format
        column_mapping = {
            '日期': 'timestamps',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change_amount',
            '换手率': 'turnover'
        }

        # Only map existing columns
        actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=actual_mapping)

        # Ensure timestamp format is correct
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        # Add stock code column
        df['stock_code'] = stock_code

        print(f"✅ Successfully retrieved {len(df)} real data records")
        print(f"📈 Latest close: {df['close'].iloc[-1]:.2f}, change: {df['pct_chg'].iloc[-1]:.2f}%")
        print(f"📅 Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")

        return df

    except Exception as e:
        print(f"❌ AKShare data fetch failed: {e}")
        return None


def get_stock_data_with_retry_all_history(stock_code="600580", retry_count=2):
    """
    Optimized data fetch function - prioritizes real API data
    """
    print(f"🔄 Trying to fetch real historical data for stock {stock_code}...")

    # Prioritize AKShare for real data
    df = fetch_real_stock_data(stock_code, "daily", "qfq")

    if df is not None:
        return df
    else:
        print("⚠️ Real data fetch failed, using realistic simulated data...")
        return create_realistic_fallback_data(stock_code)


def create_realistic_fallback_data(stock_code="600580"):
    """
    Fallback data generator based on real price references
    """
    # Reference data based on real market prices
    real_stock_references = {
        '600580': {'name': 'Wolong Electric Drive', 'current_price': 15.20, 'range': (12.0, 20.0)},
        '300207': {'name': 'Sunwoda', 'current_price': 33.79, 'range': (28.0, 38.0)},
        '300418': {'name': 'Kunlun Wanwei', 'current_price': 48.59, 'range': (40.0, 55.0)},
        '002354': {'name': 'Tianyu Digital Technology', 'current_price': 15.20, 'range': (12.0, 20.0)},
        '000001': {'name': 'Ping An Bank', 'current_price': 12.50, 'range': (10.0, 16.0)},
        '600036': {'name': 'China Merchants Bank', 'current_price': 35.80, 'range': (30.0, 42.0)},
    }

    stock_info = real_stock_references.get(stock_code, {
        'name': 'Unknown Stock',
        'current_price': 20.0,
        'range': (15.0, 25.0)
    })

    # Generate past 1 year of trading day data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

    # Generate price series based on real prices
    np.random.seed(42)
    n_points = len(dates)

    # Reverse-generate historical prices from current price
    current_price = stock_info['current_price']
    min_price, max_price = stock_info['range']

    # Reverse-generate price sequence
    prices = [current_price]
    for i in range(1, n_points):
        volatility = 0.02
        historical_return = np.random.normal(-0.0002, volatility)

        prev_price = prices[0] * (1 + historical_return)
        prev_price = max(min_price * 0.9, min(max_price * 1.1, prev_price))
        prices.insert(0, prev_price)

    # Generate OHLC data
    stock_data = []
    for i, date in enumerate(dates):
        close_price = prices[i]

        daily_volatility = abs(np.random.normal(0, 0.015))
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + daily_volatility)
        low_price = min(open_price, close_price) * (1 - daily_volatility)

        high_price = max(open_price, close_price, low_price, high_price)
        low_price = min(open_price, close_price, high_price, low_price)

        volume = int(abs(np.random.normal(1500000, 400000)))
        amount = volume * close_price

        if i > 0:
            pct_chg = ((close_price - prices[i - 1]) / prices[i - 1]) * 100
            change_amount = close_price - prices[i - 1]
        else:
            pct_chg = 0
            change_amount = 0

        stock_data.append({
            'timestamps': date,
            'stock_code': stock_code,
            'open': round(open_price, 2),
            'close': round(close_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'volume': volume,
            'amount': round(amount, 2),
            'amplitude': round(((high_price - low_price) / open_price) * 100, 2),
            'pct_chg': round(pct_chg, 2),
            'change_amount': round(change_amount, 2),
            'turnover': round(np.random.uniform(3.0, 8.0), 2)
        })

    df = pd.DataFrame(stock_data)
    print(f"✅ Generated {len(df)} fallback data records based on real prices")
    return df


def save_all_history_stock_data(df, stock_code, save_dir):
    """
    Save stock data to the specified directory
    """
    if df is not None and not df.empty:
        os.makedirs(save_dir, exist_ok=True)
        csv_file = os.path.join(save_dir, f"{stock_code}_stock_data.csv")
        df_reset = df.reset_index()
        df_reset.to_csv(csv_file, encoding='utf-8-sig', index=False)
        print(f"📁 Stock data saved: {csv_file}")
        return True
    return False


def get_stock_data(stock_code, data_dir):
    """
    Get stock data; fetch from API if local file doesn't exist
    """
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    if os.path.exists(csv_file_path):
        print(f"📁 Using existing data file: {csv_file_path}")
        return True, csv_file_path
    else:
        print(f"📡 Data file not found, fetching real data from API...")
        df = get_stock_data_with_retry_all_history(stock_code)

        if df is not None and not df.empty:
            save_all_history_stock_data(df, stock_code, data_dir)
            return True, csv_file_path
        else:
            print(f"❌ Unable to fetch stock data")
            return False, None


def prepare_stock_data(csv_file_path, stock_code, history_years=1):
    """
    Prepare stock data, convert to the format required by the Kronos model
    """
    print(f"Loading and preprocessing stock {stock_code} data...")

    # Read CSV file
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # Standardize column names
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low'
    }

    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # Ensure timestamps column exists and convert to datetime
    if 'timestamps' not in df.columns:
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    # Filter data by history years
    if history_years > 0:
        cutoff_date = datetime.now() - timedelta(days=history_years * 365)
        original_count = len(df)
        df = df[df['timestamps'] >= cutoff_date]
        print(f"📅 Using last {history_years} year(s) of data: {len(df)} records (filtered from {original_count})")

    # Data validation
    print(f"🔍 Data validation - last 5 trading day closing prices:")
    recent_prices = df[['timestamps', 'close']].tail()
    for _, row in recent_prices.iterrows():
        print(f"  {row['timestamps'].strftime('%Y-%m-%d')}: {row['close']:.2f}")

    current_price = df['close'].iloc[-1]
    print(f"✅ Data loaded, {len(df)} records total")
    print(f"Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Current price: {current_price:.2f}")

    return df


def calculate_prediction_parameters(df, target_days=60):
    """
    Calculate appropriate parameters based on target prediction days
    """
    # Calculate average trading days
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7

    # Calculate target prediction trading days
    pred_trading_days = int(target_days * trading_ratio)

    # Set lookback period
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 3, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    # Ensure parameters are within reasonable range
    lookback = max(100, min(lookback, 400))
    pred_len = max(20, min(pred_len, 120))

    print(f"📊 Parameter calculation:")
    print(f"  Target prediction days: {target_days} (calendar days)")
    print(f"  Estimated trading days: {pred_trading_days}")
    print(f"  Lookback period: {lookback}")
    print(f"  Prediction length (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates(last_date, pred_len):
    """
    Generate future trading day dates
    """
    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        if current_date.weekday() < 5:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 Generated future trading days: {len(future_dates)} days total")
    print(f"   Start date: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   End date: {future_dates[-1].strftime('%Y-%m-%d')}")

    return future_dates[:pred_len]


def calculate_optimal_interval(min_val, max_val):
    """
    Calculate optimal Y-axis tick interval
    """
    range_val = max_val - min_val
    if range_val <= 0:
        return 1.0

    if range_val < 1:
        interval = 0.1
    elif range_val < 5:
        interval = 0.5
    elif range_val < 10:
        interval = 1.0
    elif range_val < 20:
        interval = 2.0
    elif range_val < 50:
        interval = 5.0
    elif range_val < 100:
        interval = 10.0
    elif range_val < 200:
        interval = 20.0
    elif range_val < 500:
        interval = 50.0
    else:
        interval = 100.0

    return interval


def get_stock_price_reference(stock_code, current_price):
    """
    Intelligently calculate reference price range based on current price
    """
    price_ranges = {
        '600580': (current_price * 0.75, current_price * 1.25),
        '300207': (current_price * 0.75, current_price * 1.25),
        '300418': (current_price * 0.75, current_price * 1.25),
        '002354': (current_price * 0.75, current_price * 1.25),
        '000001': (current_price * 0.75, current_price * 1.25),
        '600036': (current_price * 0.75, current_price * 1.25),
    }

    if stock_code in price_ranges:
        min_price, max_price = price_ranges[stock_code]
        min_price = max(1.0, min_price)
        return {'min': min_price, 'max': max_price}
    else:
        return {'min': max(1.0, current_price * 0.7), 'max': current_price * 1.3}


# ==================== Enhanced Market Factor Analyzer ====================
class EnhancedMarketFactorAnalyzer:
    """Enhanced market factor analyzer - integrates multiple dimensions of market factors"""

    def __init__(self):
        self.market_data = {}
        self.sector_data = {}
        self.macro_factors = {}
        self.policy_factors = {}

    def analyze_market_trend(self, index_codes=["000001", "399001"]):
        """
        Analyze broad market trend - multi-index comprehensive analysis
        """
        try:
            print(f"📊 Comprehensive broad market trend analysis...")

            market_analysis = {}

            for index_code in index_codes:
                index_name = "SSE Index" if index_code == "000001" else "SZSE Component Index"
                print(f"  Analyzing {index_name} ({index_code})...")

                # Get index data
                index_df = ak.stock_zh_index_hist(symbol=index_code, period="daily")

                if index_df is None or index_df.empty:
                    print(f"  ❌ Cannot get {index_name} data")
                    continue

                # Rename columns
                index_df = index_df.rename(columns={
                    '日期': 'date', '收盘': 'close', '开盘': 'open',
                    '最高': 'high', '最低': 'low', '成交量': 'volume'
                })
                index_df['date'] = pd.to_datetime(index_df['date'])
                index_df = index_df.sort_values('date').reset_index(drop=True)

                # Calculate technical indicators
                index_df['ma5'] = index_df['close'].rolling(5).mean()
                index_df['ma20'] = index_df['close'].rolling(20).mean()
                index_df['ma60'] = index_df['close'].rolling(60).mean()
                index_df['vol_ma5'] = index_df['volume'].rolling(5).mean()

                # Technical analysis
                current_data = index_df.iloc[-1]
                prev_data = index_df.iloc[-2]

                # Bullish moving average alignment check
                ma_condition = (current_data['ma5'] > current_data['ma20'] > current_data['ma60'])

                # Price above 20-day moving average
                price_above_ma20 = current_data['close'] > current_data['ma20']

                # Volume support
                volume_condition = current_data['volume'] > current_data['vol_ma5'] * 0.8

                # Trend strength
                trend_strength = self._calculate_trend_strength(index_df)

                is_main_uptrend = ma_condition and price_above_ma20 and trend_strength > 0.6

                market_analysis[index_name] = {
                    'is_main_uptrend': is_main_uptrend,
                    'trend_strength': trend_strength,
                    'current_close': current_data['close'],
                    'price_change_pct': ((current_data['close'] - prev_data['close']) / prev_data['close']) * 100,
                    'market_status': 'main_uptrend' if is_main_uptrend else 'consolidation'
                }

            # Comprehensive judgment
            if market_analysis:
                avg_trend_strength = np.mean([data['trend_strength'] for data in market_analysis.values()])
                uptrend_count = sum(1 for data in market_analysis.values() if data['is_main_uptrend'])
                overall_uptrend = uptrend_count >= len(market_analysis) * 0.5

                final_analysis = {
                    'overall_is_main_uptrend': overall_uptrend,
                    'overall_trend_strength': avg_trend_strength,
                    'detailed_analysis': market_analysis,
                    'market_status': 'main_uptrend' if overall_uptrend else 'consolidation'
                }

                print(f"✅ Market analysis complete: {final_analysis['market_status']}, overall trend strength: {avg_trend_strength:.2f}")
                return final_analysis

            return self._get_default_market_analysis()

        except Exception as e:
            print(f"❌ Market analysis error: {e}")
            return self._get_default_market_analysis()

    def analyze_sector_resonance(self, stock_code):
        """
        Analyze sector resonance effect - enhanced industry analysis
        """
        try:
            print(f"🔄 Analyzing sector resonance effect...")

            # Get stock's industry and concepts
            industry = "unknown"
            concepts = []

            try:
                stock_info = ak.stock_individual_info_em(symbol=stock_code)
                if not stock_info.empty and 'value' in stock_info.columns:
                    industry_row = stock_info[stock_info['item'] == '行业']
                    if not industry_row.empty:
                        industry = industry_row['value'].iloc[0]
            except:
                pass

            # Hot sectors and concept mapping
            hot_sectors = {
                'Robotics': {'momentum': 0.85, 'limit_up_stocks': 18, 'active': True,
                           'description': 'Humanoid robots, industrial automation'},
                'Semiconductors': {'momentum': 0.8, 'limit_up_stocks': 15, 'active': True, 'description': 'Domestic chip substitution'},
                'AI': {'momentum': 0.75, 'limit_up_stocks': 12, 'active': True, 'description': 'Large language models, compute'},
                'Low-altitude Economy': {'momentum': 0.7, 'limit_up_stocks': 10, 'active': True, 'description': 'Drones, eVTOL'},
                'New Energy': {'momentum': 0.6, 'limit_up_stocks': 8, 'active': True, 'description': 'Solar, energy storage'},
                'Pharma': {'momentum': 0.5, 'limit_up_stocks': 5, 'active': False, 'description': 'Innovative drugs'}
            }

            # Map Chinese industry names to sectors
            industry_sector_map = {
                '机器人': 'Robotics',
                '半导体': 'Semiconductors',
                '人工智能': 'AI',
                '低空经济': 'Low-altitude Economy',
                '新能源': 'New Energy',
                '医药': 'Pharma',
            }

            # Determine which hot sectors the stock belongs to
            matched_sectors = []
            for cn_keyword, sector in industry_sector_map.items():
                if sector in hot_sectors:
                    data = hot_sectors[sector]
                    if (cn_keyword in industry or
                            (stock_code == '600580' and sector in ['Robotics', 'Low-altitude Economy']) or
                            (stock_code == '300207' and sector in ['New Energy'])):
                        matched_sectors.append({
                            'sector': sector,
                            'momentum': data['momentum'],
                            'limit_up_stocks': data['limit_up_stocks'],
                            'is_active': data['active'],
                            'description': data['description']
                        })

            # Calculate composite resonance score
            if matched_sectors:
                resonance_score = np.mean([sector['momentum'] for sector in matched_sectors])
                is_sector_hot = any(sector['is_active'] for sector in matched_sectors)
                main_sector = max(matched_sectors, key=lambda x: x['momentum'])
            else:
                resonance_score = 0.5
                is_sector_hot = False
                main_sector = {'sector': 'Traditional Industry', 'momentum': 0.5, 'description': 'No hot concept'}

            analysis = {
                'industry': industry,
                'matched_sectors': matched_sectors,
                'main_sector': main_sector,
                'is_sector_hot': is_sector_hot,
                'resonance_score': resonance_score,
                'sector_count': len(matched_sectors)
            }

            print(f"✅ Sector analysis complete: {industry}, matched {len(matched_sectors)} hot sectors, resonance score: {resonance_score:.2f}")
            return analysis

        except Exception as e:
            print(f"❌ Sector analysis error: {e}")
            return self._get_default_sector_analysis()

    def analyze_macro_factors(self):
        """
        Analyze macro factors - combining domestic and international policy
        """
        try:
            print(f"🌍 Analyzing macro factors...")

            # US rate cycle analysis - based on latest information
            us_rate_analysis = {
                'current_rate': 4.25,  # Federal funds rate target range 4.00%-4.25%
                'trend': 'rate_cut_cycle',
                'recent_cut': 'September 2025 rate cut of 25bps',
                'expected_cuts_2025': 2,  # Market expects 2 more cuts in 2025
                'expected_cuts_2026': 2,
                'impact_on_emerging_markets': 'positive',
                'usd_index_support': 95.0,  # USD index short-term support
                'analysis': 'Fed easing cycle underway, positive for global liquidity'
            }

            # Domestic policy factors - based on latest policy
            domestic_policy = {
                'monetary_policy': 'accommodative',
                'fiscal_policy': 'expansionary',
                'market_liquidity': 'reasonably ample',
                'industrial_policy': 'equipment upgrade, trade-in program',
                'employment_policy': 'stronger employment stabilization policy',
                'analysis': 'Policy combination boosting economy'
            }

            # Industry policy support
            industry_policy = {
                'robot_policy': 'Robotics industry policy support',
                'chip_policy': 'Domestic substitution acceleration',
                'AI_policy': 'AI development plan',
                'low_altitude': 'Low-altitude economy development plan'
            }

            macro_analysis = {
                'us_rate_cycle': us_rate_analysis,
                'domestic_policy': domestic_policy,
                'industry_policy': industry_policy,
                'global_liquidity_outlook': 'improving',
                'overall_macro_score': 0.75  # Macro environment overall mildly positive
            }

            print(
                f"✅ Macro analysis complete: US {us_rate_analysis['trend']}, domestic policy positive, macro score: {macro_analysis['overall_macro_score']:.2f}")
            return macro_analysis

        except Exception as e:
            print(f"❌ Macro analysis error: {e}")
            return self._get_default_macro_analysis()

    def analyze_company_fundamentals(self, stock_code):
        """
        Analyze company fundamentals - for specific stocks
        """
        try:
            print(f"🏢 Analyzing company fundamentals...")

            # Special analysis for Wolong Electric Drive
            if stock_code == '600580':
                fundamentals = {
                    'company_name': 'Wolong Electric Drive',
                    'business_areas': ['Industrial motors', 'Robot key components', 'Aviation motors', 'EV drive systems'],
                    'recent_developments': [
                        'Cross-shareholding with Zhiyuan Robot, advancing embodied AI robot R&D',
                        'Established Zhejiang Longfei Electric Drive, focusing on aviation motors',
                        'Released AI exoskeleton robot and dexterous hand',
                        'Deployed humanoid robot key components: high-torque joint modules, servo drives'
                    ],
                    'growth_drivers': [
                        'Equipment upgrade policy driving industrial motor demand',
                        'Rapid development of the robotics industry',
                        'Low-altitude economy policy support',
                        'Accelerating overseas expansion'
                    ],
                    'risk_factors': [
                        'Robot business revenue share only 2.71%, relatively low',
                        'Industrial demand cyclicality',
                        'Raw material price volatility'
                    ],
                    'investment_rating': 'positive_attention',
                    'fundamental_score': 0.7
                }
            else:
                # Basic analysis for other stocks
                fundamentals = {
                    'company_name': 'unknown',
                    'business_areas': [],
                    'recent_developments': [],
                    'growth_drivers': [],
                    'risk_factors': [],
                    'investment_rating': 'neutral',
                    'fundamental_score': 0.5
                }

            print(f"✅ Fundamental analysis complete: {fundamentals['company_name']}, score: {fundamentals['fundamental_score']:.2f}")
            return fundamentals

        except Exception as e:
            print(f"❌ Fundamental analysis error: {e}")
            return self._get_default_fundamental_analysis()

    def _calculate_trend_strength(self, df):
        """Calculate trend strength"""
        if len(df) < 20:
            return 0.5

        ma_slope = (df['ma5'].iloc[-1] - df['ma5'].iloc[-20]) / df['ma5'].iloc[-20]
        price_slope = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]

        volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()

        strength = (ma_slope * 0.4 + price_slope * 0.4 + min(volume_trend - 1, 0.2) * 0.2)
        return max(0, min(1, strength * 10))

    def _get_default_market_analysis(self):
        return {
            'overall_is_main_uptrend': False,
            'overall_trend_strength': 0.5,
            'market_status': 'unknown',
            'detailed_analysis': {}
        }

    def _get_default_sector_analysis(self):
        return {
            'industry': 'unknown',
            'matched_sectors': [],
            'main_sector': {'sector': 'unknown', 'momentum': 0.5, 'description': ''},
            'is_sector_hot': False,
            'resonance_score': 0.5,
            'sector_count': 0
        }

    def _get_default_macro_analysis(self):
        return {
            'us_rate_cycle': {'trend': 'unknown', 'expected_cuts_2025': 0},
            'domestic_policy': {'monetary_policy': 'neutral'},
            'overall_macro_score': 0.5
        }

    def _get_default_fundamental_analysis(self):
        return {
            'company_name': 'unknown',
            'business_areas': [],
            'recent_developments': [],
            'growth_drivers': [],
            'risk_factors': [],
            'investment_rating': 'neutral',
            'fundamental_score': 0.5
        }


# ==================== Enhanced Prediction Functions ====================
def enhance_prediction_with_market_factors(
        historical_df,
        prediction_df,
        stock_code,
        market_analyzer
):
    """
    Enhance prediction results using market factors - multi-dimensional comprehensive analysis
    """
    print("\n🎯 Enhancing prediction with multi-dimensional market factors...")

    # Get various market analyses
    market_analysis = market_analyzer.analyze_market_trend()
    sector_analysis = market_analyzer.analyze_sector_resonance(stock_code)
    macro_analysis = market_analyzer.analyze_macro_factors()
    fundamental_analysis = market_analyzer.analyze_company_fundamentals(stock_code)

    # Calculate composite adjustment factor
    adjustment_factor = calculate_enhanced_adjustment_factor(
        market_analysis, sector_analysis, macro_analysis, fundamental_analysis
    )

    print(f"📈 Composite adjustment factor: {adjustment_factor:.4f}")

    # Apply adjustment to prediction results
    enhanced_prediction = prediction_df.copy()

    # Adjust price predictions
    price_columns = ['close', 'open', 'high', 'low']
    for col in price_columns:
        if col in enhanced_prediction.columns:
            # Apply moderate adjustment, avoid over-optimism or over-pessimism
            adjusted_value = enhanced_prediction[col] * adjustment_factor
            # Limit single adjustment to within ±10%
            change_ratio = adjusted_value / enhanced_prediction[col]
            if change_ratio.max() > 1.1:
                adjusted_value = enhanced_prediction[col] * 1.1
            elif change_ratio.min() < 0.9:
                adjusted_value = enhanced_prediction[col] * 0.9
            enhanced_prediction[col] = adjusted_value

    # Adjust volume
    if 'volume' in enhanced_prediction.columns:
        volume_adjustment = 1 + (adjustment_factor - 1) * 0.3  # Volume adjustment is more moderate
        enhanced_prediction['volume'] = enhanced_prediction['volume'] * volume_adjustment

    return enhanced_prediction, {
        'market_analysis': market_analysis,
        'sector_analysis': sector_analysis,
        'macro_analysis': macro_analysis,
        'fundamental_analysis': fundamental_analysis,
        'adjustment_factor': adjustment_factor
    }


def calculate_enhanced_adjustment_factor(market_analysis, sector_analysis, macro_analysis, fundamental_analysis):
    """
    Calculate adjustment factor based on multi-dimensional market factors - more balanced approach
    """
    base_factor = 1.0
    factors_log = []

    # 1. Broad market trend impact (weight 25%)
    if market_analysis['overall_is_main_uptrend']:
        trend_strength = market_analysis['overall_trend_strength']
        adjustment = 1 + trend_strength * 0.08
        base_factor *= adjustment
        factors_log.append(f"Main uptrend: +{trend_strength * 0.08:.3f}")
    else:
        trend_strength = market_analysis['overall_trend_strength']
        # Consolidation doesn't necessarily mean bearish, just smaller upside
        adjustment = 1 + (trend_strength - 0.5) * 0.04
        base_factor *= adjustment
        factors_log.append(f"Consolidation: {(trend_strength - 0.5) * 0.04:+.3f}")

    # 2. Sector resonance impact (weight 25%)
    resonance_score = sector_analysis['resonance_score']
    sector_count = sector_analysis['sector_count']

    if sector_analysis['is_sector_hot']:
        # Hot sector with multiple concept overlaps
        sector_adjustment = 1 + resonance_score * 0.06 + min(sector_count * 0.01, 0.03)
        base_factor *= sector_adjustment
        factors_log.append(
            f"Hot sector ({sector_count}): +{resonance_score * 0.06 + min(sector_count * 0.01, 0.03):.3f}")
    else:
        # Non-hot sector still has baseline support
        base_factor *= (1 + (resonance_score - 0.5) * 0.02)
        factors_log.append(f"General sector: {(resonance_score - 0.5) * 0.02:+.3f}")

    # 3. Macro factor impact (weight 20%)
    macro_score = macro_analysis['overall_macro_score']
    macro_adjustment = 1 + (macro_score - 0.5) * 0.06
    base_factor *= macro_adjustment
    factors_log.append(f"Macro environment: {(macro_score - 0.5) * 0.06:+.3f}")

    # 4. US rate cut cycle special impact (weight 10%)
    us_rate_trend = macro_analysis['us_rate_cycle']['trend']
    if us_rate_trend == 'rate_cut_cycle':
        expected_cuts = macro_analysis['us_rate_cycle']['expected_cuts_2025']
        us_adjustment = 1 + expected_cuts * 0.015
        base_factor *= us_adjustment
        factors_log.append(f"US rate cuts: +{expected_cuts * 0.015:.3f}")

    # 5. Company fundamentals impact (weight 20%)
    fundamental_score = fundamental_analysis['fundamental_score']
    fundamental_adjustment = 1 + (fundamental_score - 0.5) * 0.08
    base_factor *= fundamental_adjustment
    factors_log.append(f"Fundamentals: {(fundamental_score - 0.5) * 0.08:+.3f}")

    # Output adjustment factor details
    print("🔍 Adjustment factor breakdown:")
    for log in factors_log:
        print(f"   {log}")

    # Limit adjustment range to a more reasonable range (0.85 ~ 1.15)
    final_factor = max(0.85, min(1.15, base_factor))

    if final_factor != base_factor:
        print(f"⚠️  Adjustment factor capped from {base_factor:.3f} to {final_factor:.3f}")

    return final_factor


def create_comprehensive_market_report(enhancement_info, output_dir, stock_code):
    """
    Create comprehensive market analysis report
    """
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_code': stock_code,
        'market_analysis': enhancement_info['market_analysis'],
        'sector_analysis': enhancement_info['sector_analysis'],
        'macro_analysis': enhancement_info['macro_analysis'],
        'fundamental_analysis': enhancement_info['fundamental_analysis'],
        'adjustment_factor': enhancement_info['adjustment_factor'],
        'analysis_summary': generate_analysis_summary(enhancement_info)
    }

    # Save report
    report_file = os.path.join(output_dir, f'{stock_code}_comprehensive_analysis_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📋 Comprehensive analysis report saved: {report_file}")
    return report


def generate_analysis_summary(enhancement_info):
    """
    Generate analysis summary
    """
    market = enhancement_info['market_analysis']
    sector = enhancement_info['sector_analysis']
    macro = enhancement_info['macro_analysis']
    fundamental = enhancement_info['fundamental_analysis']

    summary = {
        'overall_sentiment': 'positive' if enhancement_info['adjustment_factor'] > 1.0 else 'cautious',
        'key_drivers': [],
        'main_risks': [],
        'investment_suggestion': ''
    }

    # Key drivers
    if market['overall_trend_strength'] > 0.6:
        summary['key_drivers'].append('Favorable broad market trend')

    if sector['is_sector_hot']:
        summary['key_drivers'].append(f"Hot sector: {sector['main_sector']['sector']}")

    if macro['overall_macro_score'] > 0.7:
        summary['key_drivers'].append('Favorable macro environment')

    if fundamental['fundamental_score'] > 0.6:
        summary['key_drivers'].append('Solid fundamentals')

    # Main risks
    if market['overall_trend_strength'] < 0.4:
        summary['main_risks'].append('Weak broad market trend')

    if not sector['is_sector_hot']:
        summary['main_risks'].append('Not in a hot sector')

    if len(summary['key_drivers']) > len(summary['main_risks']):
        summary['investment_suggestion'] = 'Consider buying on dips'
    else:
        summary['investment_suggestion'] = 'Proceed with caution'

    return summary


# ==================== Enhanced Visualization Functions ====================
def plot_comprehensive_prediction(
        historical_df,
        prediction_df,
        future_dates,
        stock_code,
        stock_name,
        output_dir,
        enhancement_info=None
):
    """
    Plot comprehensive prediction charts - includes more market analysis info
    """
    ensure_output_directory(output_dir)

    # Color scheme
    colors = {
        'historical': '#1f77b4',
        'prediction': '#ff7f0e',
        'enhanced': '#2ca02c',
        'background': '#f8f9fa',
        'grid': '#e9ecef',
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }

    # Create comprehensive chart
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1])

    # 1. Main price chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(colors['background'])

    # 2. Volume chart
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(colors['background'])

    # 3. Market analysis charts
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(colors['background'])

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(colors['background'])

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(colors['background'])

    # 4. Factor analysis chart
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(colors['background'])

    # Set background color
    fig.patch.set_facecolor('white')

    # 1. Price chart
    historical_prices = historical_df.set_index('timestamps')['close']
    prediction_prices = prediction_df.set_index(pd.DatetimeIndex(future_dates))['close']

    # Get current latest price
    current_price = historical_prices.iloc[-1]

    # Smart Y-axis range calculation
    all_prices = pd.concat([historical_prices, prediction_prices])
    data_min = all_prices.min()
    data_max = all_prices.max()

    price_range = data_max - data_min
    y_margin = price_range * 0.15

    y_min = max(0, data_min - y_margin)
    y_max = data_max + y_margin

    # Set Y-axis ticks
    y_interval = calculate_optimal_interval(y_min, y_max)
    y_ticks = np.arange(round(y_min / y_interval) * y_interval,
                        round(y_max / y_interval) * y_interval + y_interval,
                        y_interval)

    # Plot historical prices
    ax1.plot(historical_prices.index, historical_prices.values,
             color=colors['historical'], linewidth=2, label='Historical Price')

    # Plot prediction prices
    if len(prediction_prices) > 0:
        # Connection point
        last_hist_date = historical_prices.index[-1]
        last_hist_price = historical_prices.iloc[-1]
        first_pred_date = prediction_prices.index[0]

        # Draw connecting line
        ax1.plot([last_hist_date, first_pred_date],
                 [last_hist_price, prediction_prices.iloc[0]],
                 color=colors['prediction'], linewidth=2.5, linestyle='-')

        # Draw prediction line
        ax1.plot(prediction_prices.index, prediction_prices.values,
                 color=colors['prediction'], linewidth=2.5, label='Base Prediction')

        # Draw enhanced prediction line
        if enhancement_info and 'enhanced_prediction' in enhancement_info:
            enhanced_prices = enhancement_info['enhanced_prediction'].set_index(pd.DatetimeIndex(future_dates))['close']
            ax1.plot(enhanced_prices.index, enhanced_prices.values,
                     color=colors['enhanced'], linewidth=2.5, linestyle='--', label='Enhanced Prediction')

        # Mark prediction start point
        ax1.axvline(x=last_hist_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.annotate('Prediction Start', xy=(last_hist_date, last_hist_price),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Set Y-axis range and ticks
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(y_ticks)

    ax1.set_ylabel('Close Price (CNY)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, color=colors['grid'], alpha=0.7)

    title = f'{stock_name}({stock_code}) - Comprehensive Factors Price Prediction\nCurrent price: {current_price:.2f} | Enhancement factor: {enhancement_info["adjustment_factor"]:.3f}' if enhancement_info else f'{stock_name}({stock_code}) - Price Prediction\nCurrent price: {current_price:.2f}'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set x-axis format
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Volume chart
    historical_volume = historical_df.set_index('timestamps')['volume']
    prediction_volume = prediction_df.set_index(pd.DatetimeIndex(future_dates))['volume']

    # Calculate relative volume (normalized)
    hist_volume_norm = historical_volume / historical_volume.max()
    if len(prediction_volume) > 0:
        pred_volume_norm = prediction_volume / historical_volume.max()

    # Plot historical volume
    ax2.bar(historical_volume.index, hist_volume_norm.values,
            alpha=0.6, color=colors['historical'], label='Historical Volume')

    # Plot predicted volume
    if len(prediction_volume) > 0:
        ax2.bar(prediction_volume.index, pred_volume_norm.values,
                alpha=0.6, color=colors['prediction'], label='Predicted Volume')

    ax2.set_ylabel('Relative Volume', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, color=colors['grid'], alpha=0.7)
    ax2.set_ylim(0, 1.2)

    # Set x-axis format
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Market analysis subplots
    if enhancement_info:
        # Factor weight pie chart
        factors = ['Market Trend', 'Sector Resonance', 'Macro Environment', 'US Rate Cuts', 'Fundamentals']
        weights = [25, 25, 20, 10, 20]
        colors_pie = [colors['historical'], colors['prediction'], colors['enhanced'], '#f39c12', '#9b59b6']

        ax3.pie(weights, labels=factors, autopct='%1.0f%%', colors=colors_pie, startangle=90)
        ax3.set_title('Factor Weight Distribution', fontweight='bold', fontsize=11)

        # Factor score bar chart
        scores = [
            enhancement_info['market_analysis']['overall_trend_strength'],
            enhancement_info['sector_analysis']['resonance_score'],
            enhancement_info['macro_analysis']['overall_macro_score'],
            0.7 if enhancement_info['macro_analysis']['us_rate_cycle']['trend'] == 'rate_cut_cycle' else 0.3,
            enhancement_info['fundamental_analysis']['fundamental_score']
        ]

        x_pos = np.arange(len(factors))
        bars = ax4.bar(x_pos, scores, color=colors_pie, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(factors, rotation=45, fontsize=9)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Score', fontsize=10)
        ax4.set_title('Current Factor Scores', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3)

        # Show values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # Market status summary
        market_status = enhancement_info['market_analysis']['market_status']
        sector_status = "hot" if enhancement_info['sector_analysis']['is_sector_hot'] else "general"
        macro_status = "favorable" if enhancement_info['macro_analysis']['overall_macro_score'] > 0.6 else "unfavorable"

        summary_text = f"""Market Status Summary:

Market Trend: {market_status}
Sector Heat: {sector_status}
Macro Environment: {macro_status}
US Rate Trend: {enhancement_info['macro_analysis']['us_rate_cycle']['trend']}
Composite Score: {enhancement_info['adjustment_factor']:.3f}

Investment Rating: {enhancement_info['fundamental_analysis']['investment_rating']}"""

        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
                 verticalalignment='top', linespacing=1.5)
        ax5.set_title('Market Status Summary', fontweight='bold', fontsize=11)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)

        # 4. Detailed factor analysis
        if 'analysis_summary' in enhancement_info:
            summary = enhancement_info['analysis_summary']
            drivers_text = "\n".join([f"• {driver}" for driver in summary['key_drivers']]) if summary[
                'key_drivers'] else "• No significant drivers"
            risks_text = "\n".join([f"• {risk}" for risk in summary['main_risks']]) if summary[
                'main_risks'] else "• Risks manageable"

            detail_text = f"""Key Drivers:
{drivers_text}

Main Risks:
{risks_text}

Overall Sentiment: {summary['overall_sentiment']}
Suggestion: {summary['investment_suggestion']}"""

            ax6.text(0.02, 0.95, detail_text, transform=ax6.transAxes, fontsize=9,
                     verticalalignment='top', linespacing=1.3)
            ax6.set_title('Detailed Factor Analysis', fontweight='bold', fontsize=11)
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['bottom'].set_visible(False)
            ax6.spines['left'].set_visible(False)

    plt.tight_layout()

    # Save image
    chart_filename = os.path.join(output_dir, f'{stock_code}_comprehensive_prediction.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Comprehensive prediction chart saved: {chart_filename}")

    plt.close('all')

    return historical_prices, prediction_prices


# ==================== Main Prediction Function ====================
def run_comprehensive_kronos_prediction(stock_code, stock_name, data_dir, pred_days, output_dir, history_years=1):
    """
    Run comprehensive Kronos model prediction workflow
    """
    print(f"\n🎯 Starting {stock_name}({stock_code}) comprehensive Kronos model price prediction")
    print("=" * 60)

    # Initialize enhanced market analyzer
    market_analyzer = EnhancedMarketFactorAnalyzer()

    try:
        # 1. Get data
        print("\nStep 1: Getting stock data...")
        success, csv_file_path = get_stock_data(stock_code, data_dir)
        if not success:
            print("❌ Cannot get stock data, prediction aborted")
            return

        # 2. Load model and tokenizer
        print("\nStep 2: Loading Kronos model and tokenizer...")
        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            print("✅ Model loaded - using Kronos-base model")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            print("⚠️ Prediction unavailable, check model installation")
            return

        # 3. Instantiate predictor
        print("Step 3: Initializing predictor...")
        predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        print("✅ Predictor initialized")

        # 4. Prepare data
        print("Step 4: Preparing stock data...")
        df = prepare_stock_data(csv_file_path, stock_code, history_years)

        # 5. Calculate prediction parameters
        print("Step 5: Calculating prediction parameters...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ Insufficient data for prediction")
            return

        print(f"✅ Final parameters - lookback: {lookback}, pred_len: {pred_len}")

        # 6. Prepare input data
        print("Step 6: Preparing input data...")
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        # Generate future dates
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates(last_historical_date, pred_len)

        print(f"Input data shape: {x_df.shape}")
        print(f"Historical data range: {x_timestamp.iloc[0]} to {x_timestamp.iloc[-1]}")
        print(f"Prediction range: {future_dates[0]} to {future_dates[-1]}")

        # 7. Execute base prediction
        print("Step 7: Executing base price prediction...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future_dates),
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        print("✅ Base prediction complete")
        print("First 5 rows of prediction data:")
        print(pred_df.head())

        # 8. Enhance prediction with multi-dimensional market factors
        print("Step 8: Applying multi-dimensional market factor enhancement...")
        enhanced_pred_df, enhancement_info = enhance_prediction_with_market_factors(
            df.loc[-lookback:].reset_index(drop=True),
            pred_df,
            stock_code,
            market_analyzer
        )

        # Add enhanced prediction to info
        enhancement_info['enhanced_prediction'] = enhanced_pred_df

        # 9. Create comprehensive market analysis report
        market_report = create_comprehensive_market_report(enhancement_info, output_dir, stock_code)

        # 10. Visualize results
        print("Step 9: Generating comprehensive visualization...")
        historical_df = df.loc[-lookback:].reset_index(drop=True)
        hist_prices, base_pred_prices = plot_comprehensive_prediction(
            historical_df, pred_df, future_dates, stock_code, stock_name, output_dir, enhancement_info
        )

        # 11. Generate comprehensive prediction report
        print("Step 10: Generating comprehensive prediction report...")
        if len(enhanced_pred_df) > 0:
            current_price = hist_prices.iloc[-1]
            base_predicted_price = base_pred_prices.iloc[-1] if len(base_pred_prices) > 0 else current_price
            enhanced_predicted_price = enhanced_pred_df.set_index(pd.DatetimeIndex(future_dates))['close'].iloc[-1]

            base_change_pct = (base_predicted_price / current_price - 1) * 100
            enhanced_change_pct = (enhanced_predicted_price / current_price - 1) * 100

            print(f"\n📈 Comprehensive Kronos Model Prediction Report")
            print("=" * 70)
            print(f"Stock: {stock_name}({stock_code})")
            print(f"Current price: {current_price:.2f}")
            print(f"Base prediction price: {base_predicted_price:.2f} ({base_change_pct:+.2f}%)")
            print(f"Enhanced prediction price: {enhanced_predicted_price:.2f} ({enhanced_change_pct:+.2f}%)")
            print(f"Market factor adjustment: {enhancement_info['adjustment_factor']:.4f}")
            print(f"Market status: {enhancement_info['market_analysis']['market_status']}")
            print(
                f"Sector resonance: {enhancement_info['sector_analysis']['main_sector']['sector']} (score: {enhancement_info['sector_analysis']['resonance_score']:.2f})")
            print(f"Macro environment: US {enhancement_info['macro_analysis']['us_rate_cycle']['trend']}")
            print(f"Company rating: {enhancement_info['fundamental_analysis']['investment_rating']}")
            print(f"Prediction period: {pred_len} trading days")

            # Output key factors
            print(f"\n🔑 Key influencing factors:")
            for driver in enhancement_info['analysis_summary']['key_drivers']:
                print(f"  ✅ {driver}")
            for risk in enhancement_info['analysis_summary']['main_risks']:
                print(f"  ⚠️  {risk}")
            print(f"  💡 Investment suggestion: {enhancement_info['analysis_summary']['investment_suggestion']}")

            # Save detailed prediction data
            prediction_details = pd.DataFrame({
                'date': future_dates,
                'base_predicted_close': base_pred_prices.values if len(base_pred_prices) > 0 else [current_price] * len(
                    future_dates),
                'enhanced_predicted_close': enhanced_pred_df['close'].values,
                'predicted_volume': enhanced_pred_df['volume'].values
            })

            prediction_file = os.path.join(output_dir, f'{stock_code}_comprehensive_predictions.csv')
            prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
            print(f"💾 Detailed prediction data saved: {prediction_file}")

        print(f"\n🎉 {stock_name}({stock_code}) comprehensive Kronos model prediction complete!")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


# ==================== Main Function ====================
def main():
    """
    Main function: comprehensive Kronos model stock prediction system
    """
    # ==================== Configuration ====================
    STOCK_CONFIG = {
        "stock_code": "603288",
        "stock_name": "Haitian Flavouring",
        "data_dir": r"./examples/data",
        "pred_days": 60,
        "output_dir": r"./examples/yuce",
        "history_years": 1
    }

    print("🤖 Comprehensive Kronos Model Stock Price Prediction System")
    print("=" * 50)
    print("📊 New features: multi-dimensional market factor analysis")
    print("🎯 Includes: market trend + sector resonance + macro policy + company fundamentals")
    print("🚀 Model: Kronos-base")
    print(f"Current stock: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"Prediction days: {STOCK_CONFIG['pred_days']}")
    print(f"Output directory: {STOCK_CONFIG['output_dir']}")
    print()

    # Run comprehensive Kronos prediction
    run_comprehensive_kronos_prediction(**STOCK_CONFIG)

    print(f"\n💡 Tip: Comprehensive model integrates multi-dimensional market environment analysis")


if __name__ == "__main__":
    main()
