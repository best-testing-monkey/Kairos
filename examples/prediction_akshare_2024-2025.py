import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add project path for importing custom modules
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ensure_output_directory(output_dir):
    """Ensure output directory exists, create if not"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    return output_dir


def prepare_stock_data(csv_file_path, stock_code):
    """
    Prepare stock data in the format required by the Kronos model

    Parameters:
    csv_file_path: CSV file path
    stock_code: stock code for display

    Returns:
    df: processed DataFrame
    """
    print(f"Loading and preprocessing stock {stock_code} data...")

    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }

    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    if 'timestamps' not in df.columns:
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])

    df = df.sort_values('timestamps').reset_index(drop=True)

    print(f"✅ Data loaded: {len(df)} records")
    print(f"Time range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def calculate_prediction_parameters(df, target_days=100):
    """
    Calculate appropriate parameters based on target prediction days

    Parameters:
    df: stock data DataFrame
    target_days: target prediction days (calendar days)

    Returns:
    lookback: lookback period
    pred_len: prediction length
    """
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7

    pred_trading_days = int(target_days * trading_ratio)

    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 2, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    print(f"📊 Parameter calculation:")
    print(f"  Target prediction days: {target_days} (calendar days)")
    print(f"  Estimated trading days: {pred_trading_days}")
    print(f"  Lookback period: {lookback}")
    print(f"  Prediction length: {pred_len}")

    return lookback, pred_len


def generate_future_dates_with_holidays(last_date, pred_len):
    """
    Generate future trading dates accounting for Chinese holidays

    Parameters:
    last_date: date of last historical data point
    pred_len: prediction length

    Returns:
    future_dates: list of future trading dates
    """
    holidays_2025 = [
        # China National Day holiday (Oct 1-8, 2025)
        datetime(2025, 10, 1), datetime(2025, 10, 2), datetime(2025, 10, 3),
        datetime(2025, 10, 4), datetime(2025, 10, 5), datetime(2025, 10, 6),
        datetime(2025, 10, 7), datetime(2025, 10, 8),
    ]

    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        if current_date.weekday() < 5 and current_date not in holidays_2025:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 Generated future trading dates: {len(future_dates)} days")
    print(f"   Start date: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   End date: {future_dates[-1].strftime('%Y-%m-%d')}")

    holiday_count = sum(1 for date in holidays_2025 if date > last_date)
    print(f"   Holidays skipped: {holiday_count} days")

    return future_dates[:pred_len]


def plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code="002354", stock_name="Stock", pred_len=100,
                                 output_dir="."):
    """
    Plot detailed prediction result chart

    Parameters:
    kline_df: historical candlestick data
    pred_df: prediction data
    future_dates: list of future dates
    stock_code: stock code
    stock_name: stock name
    pred_len: prediction length
    output_dir: output directory
    """
    ensure_output_directory(output_dir)

    min_len = min(len(pred_df), len(future_dates))
    pred_df = pred_df.iloc[:min_len]
    future_dates = future_dates[:min_len]

    pred_df.index = future_dates

    sr_close = kline_df.set_index('timestamps')['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Historical'
    sr_pred_close.name = 'Predicted'

    sr_volume = kline_df.set_index('timestamps')['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Historical'
    sr_pred_volume.name = 'Predicted'

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig = plt.figure(figsize=(18, 14))

    gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # price chart
    ax2 = fig.add_subplot(gs[1])  # volume chart
    ax3 = fig.add_subplot(gs[2])  # price change chart

    recent_history = close_df['Historical'].iloc[-min(200, len(close_df['Historical'])):]
    ax1.plot(recent_history.index, recent_history.values, label='Historical Price', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax1.plot(close_df['Predicted'].index, close_df['Predicted'].values, label='Predicted Price',
             color='#ff7f0e', linewidth=2.5, linestyle='-', marker='o', markersize=3)

    prediction_start_date = close_df['Predicted'].index[0] if len(close_df['Predicted']) > 0 else close_df.index[-1]
    prediction_start_price = close_df['Historical'].iloc[-1]
    ax1.axvline(x=prediction_start_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate('Prediction Start', xy=(prediction_start_date, prediction_start_price),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_ylabel('Close Price (CNY)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{stock_name}({stock_code}) Stock Price Prediction - Next {pred_len} Trading Days',
                  fontsize=16, fontweight='bold', pad=20)

    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    pred_volumes = volume_df['Predicted'].dropna()
    if len(pred_volumes) > 0:
        ax2.bar(pred_volumes.index, pred_volumes.values,
                alpha=0.7, color='#ff7f0e', label='Predicted Volume', width=0.8)

    ax2.set_ylabel('Volume (lots)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    if len(pred_volumes) > 0:
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    if len(close_df['Predicted']) > 0:
        price_change = close_df['Predicted'] - close_df['Historical'].iloc[-1]
        colors = ['green' if x >= 0 else 'red' for x in price_change]

        bars = ax3.bar(range(len(price_change)), price_change, alpha=0.8, color=colors)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i % 10 == 0 or i == len(bars) - 1 or abs(height) > price_change.std():
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:+.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8, fontweight='bold')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    ax3.set_ylabel('Price Change (CNY)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trading Day', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    if len(price_change) > 0:
        xticks_positions = list(range(0, len(price_change), max(1, len(price_change) // 10)))
        if len(price_change) - 1 not in xticks_positions:
            xticks_positions.append(len(price_change) - 1)
        ax3.set_xticks(xticks_positions)
        ax3.set_xticklabels([f'D{i + 1}' for i in xticks_positions])

    if len(close_df['Predicted']) > 0 and not np.isnan(close_df['Historical'].iloc[-1]):
        pred_stats = {
            'Stock Code': stock_code,
            'Stock Name': stock_name,
            'Current Price': f"{close_df['Historical'].iloc[-1]:.2f} CNY",
            'Predicted End Price': f"{close_df['Predicted'].iloc[-1]:.2f} CNY",
            'Predicted Change': f"{(close_df['Predicted'].iloc[-1] / close_df['Historical'].iloc[-1] - 1) * 100:+.2f}%",
            'Predicted High': f"{close_df['Predicted'].max():.2f} CNY",
            'Predicted Low': f"{close_df['Predicted'].min():.2f} CNY",
            'Predicted Volatility': f"{close_df['Predicted'].std():.2f} CNY",
            'Prediction Start': f"{close_df['Predicted'].index[0].strftime('%Y-%m-%d')}",
            'Prediction End': f"{close_df['Predicted'].index[-1].strftime('%Y-%m-%d')}",
            'Trading Days': f"{len(close_df['Predicted'])} days"
        }

        stats_text = "\n".join([f"{k}: {v}" for k, v in pred_stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                 verticalalignment='bottom')

    plt.tight_layout()

    chart_filename = os.path.join(output_dir, f'{stock_code}_prediction_chart.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Prediction chart saved: {chart_filename}")

    plt.show()

    return close_df, volume_df


def generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code="002354", stock_name="Stock",
                               output_dir="."):
    """Generate prediction report"""
    ensure_output_directory(output_dir)

    print(f"\n{'=' * 70}")
    print(f"📊 {stock_name}({stock_code}) Stock Prediction Report")
    print(f"{'=' * 70}")

    if len(close_df['Predicted']) == 0 or np.isnan(close_df['Historical'].iloc[-1]):
        print("❌ No valid prediction data to generate report")
        return

    min_len = min(len(close_df['Predicted']), len(volume_df['Predicted']), len(future_dates))

    historical_close = close_df['Historical'].iloc[-1]
    predicted_close = close_df['Predicted'].iloc[-1]
    price_change_pct = (predicted_close / historical_close - 1) * 100

    print(f"🔮 Prediction Overview:")
    print(f"   Current Price: {historical_close:.2f} CNY")
    print(f"   Predicted End Price: {predicted_close:.2f} CNY")
    print(f"   Predicted Change: {price_change_pct:+.2f}%")
    print(f"   Prediction Period: {min_len} trading days")
    print(f"   Date Range: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

    print(f"\n📈 Price Prediction Statistics:")
    print(f"   Predicted High: {close_df['Predicted'].max():.2f} CNY")
    print(f"   Predicted Low: {close_df['Predicted'].min():.2f} CNY")
    print(f"   Predicted Average: {close_df['Predicted'].mean():.2f} CNY")
    print(f"   Price Volatility: {close_df['Predicted'].std():.2f} CNY")

    print(f"\n📊 Volume Prediction Statistics:")
    print(f"   Average Predicted Volume: {volume_df['Predicted'].mean():,.0f} lots")
    print(f"   Max Predicted Volume: {volume_df['Predicted'].max():,.0f} lots")
    print(f"   Min Predicted Volume: {volume_df['Predicted'].min():,.0f} lots")

    prediction_details = pd.DataFrame({
        'date': future_dates[:min_len],
        'predicted_close': close_df['Predicted'].values[:min_len],
        'predicted_volume': volume_df['Predicted'].values[:min_len],
        'price_change_cny': (close_df['Predicted'].values[:min_len] - historical_close),
        'price_change_pct': ((close_df['Predicted'].values[:min_len] / historical_close - 1) * 100)
    })

    prediction_file = os.path.join(output_dir, f'{stock_code}_detailed_predictions.csv')
    prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Detailed prediction data saved: {prediction_file}")


def main(stock_code="002354", stock_name="Tianyu Digital Technology", data_dir="./data", pred_days=100, output_dir="./output"):
    """
    Main function: run stock price prediction

    Parameters:
    stock_code: stock code
    stock_name: stock name
    data_dir: data directory
    pred_days: prediction days (calendar days)
    output_dir: output directory
    """
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    print(f"🎯 Starting {stock_name}({stock_code}) stock price prediction")
    print("=" * 70)
    print(f"Data file: {csv_file_path}")
    print(f"Prediction days: {pred_days} (calendar days)")
    print(f"Output directory: {output_dir}")

    if not os.path.exists(csv_file_path):
        print(f"❌ Data file not found: {csv_file_path}")
        print("Please run the data fetch script first to generate stock data files")
        return

    ensure_output_directory(output_dir)

    try:
        print("\nStep 1: Loading Kronos model and tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("✅ Model loaded")

        print("Step 2: Initializing predictor...")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        print("✅ Predictor initialized")

        print("Step 3: Preparing stock data...")
        df = prepare_stock_data(csv_file_path, stock_code)

        print("Step 4: Calculating prediction parameters...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ Insufficient data for prediction")
            return

        print(f"✅ Final parameters - lookback: {lookback}, pred_len: {pred_len}")

        print("Step 5: Preparing input data...")
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates_with_holidays(last_historical_date, pred_len)

        print(f"Input data shape: {x_df.shape}")
        print(f"Historical data range: {x_timestamp.iloc[0]} to {x_timestamp.iloc[-1]}")
        print(f"Prediction range: {future_dates[0]} to {future_dates[-1]}")

        print("Step 6: Running price prediction...")
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

        print("✅ Prediction complete")

        print("\nStep 7: Displaying prediction results...")
        print("First 5 rows of prediction data:")
        min_len = min(len(pred_df), len(future_dates))
        pred_df = pred_df.iloc[:min_len]
        pred_df.index = future_dates[:min_len]
        print(pred_df.head())

        print("Step 8: Generating visualization charts...")
        kline_df = df.loc[-lookback:].reset_index(drop=True)
        close_df, volume_df = plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code, stock_name,
                                                           pred_len, output_dir)

        print("Step 9: Generating prediction report...")
        generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code, stock_name, output_dir)

        print(f"\n🎉 {stock_name}({stock_code}) prediction complete!")
        print("Generated files:")
        print(f"  📊 {os.path.join(output_dir, stock_code + '_prediction_chart.png')} - prediction chart")
        print(f"  📋 {os.path.join(output_dir, stock_code + '_detailed_predictions.csv')} - detailed prediction data")

        if len(close_df['Predicted']) > 0 and not np.isnan(close_df['Historical'].iloc[-1]):
            print(f"\n📈 Prediction Summary:")
            historical_price = close_df['Historical'].iloc[-1]
            predicted_price = close_df['Predicted'].iloc[-1]
            change_pct = (predicted_price / historical_price - 1) * 100

            print(f"  Current Price: {historical_price:.2f} CNY")
            print(f"  Predicted Price: {predicted_price:.2f} CNY")
            print(f"  Expected Change: {change_pct:+.2f}%")
            print(f"  Prediction Period: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

            if change_pct > 10:
                print(f"  🚀 Model predicts strong bullish trend over next {pred_len} trading days (+{change_pct:.1f}%)")
            elif change_pct > 5:
                print(f"  📈 Model predicts bullish trend over next {pred_len} trading days (+{change_pct:.1f}%)")
            elif change_pct > 0:
                print(f"  ↗️ Model predicts slight uptrend over next {pred_len} trading days (+{change_pct:.1f}%)")
            elif change_pct > -5:
                print(f"  ↘️ Model predicts slight downtrend over next {pred_len} trading days ({change_pct:.1f}%)")
            elif change_pct > -10:
                print(f"  📉 Model predicts bearish trend over next {pred_len} trading days ({change_pct:.1f}%)")
            else:
                print(f"  🔻 Model predicts strong bearish trend over next {pred_len} trading days ({change_pct:.1f}%)")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    Stock prediction tool - supports multiple stocks

    Usage:
    Modify STOCK_CONFIG below to predict different stocks
    """

    # ==================== Modify stock configuration here ====================
    STOCK_CONFIG = {
        "stock_code": "300418",  # stock code
        "stock_name": "Kunlun Wanwei",  # stock name
        "data_dir": "./data",  # data directory
        "pred_days": 100,  # predict 100 calendar days
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"  # output directory
    }

    # Other stock config examples:
    # STOCK_CONFIG = {"stock_code": "000001", "stock_name": "Ping An Bank", "data_dir": "./data", "pred_days": 100, "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"}
    # STOCK_CONFIG = {"stock_code": "600036", "stock_name": "China Merchants Bank", "data_dir": "./data", "pred_days": 100, "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"}
    # STOCK_CONFIG = {"stock_code": "300750", "stock_name": "CATL", "data_dir": "./data", "pred_days": 100, "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce"}
    # =========================================================

    print("🤖 Intelligent Stock Prediction Tool")
    print("=" * 70)
    print(f"Current stock: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"Data directory: {STOCK_CONFIG['data_dir']}")
    print(f"Prediction days: {STOCK_CONFIG['pred_days']} (calendar days)")
    print(f"Output directory: {STOCK_CONFIG['output_dir']}")
    print()

    main(**STOCK_CONFIG)

    print(f"\n💡 Tip: Modify STOCK_CONFIG in the code to predict different stocks")
