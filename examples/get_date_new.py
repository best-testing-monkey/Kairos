import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import time
import random


def get_stock_market(stock_code):
    """
    Determine market type based on stock code.
    Returns: market prefix '0' = Shenzhen (SZSE), '1' = Shanghai (SSE)
    """
    if stock_code.startswith(('0', '2', '3')):
        return '0'  # Shenzhen Stock Exchange
    elif stock_code.startswith(('6', '9')):
        return '1'  # Shanghai Stock Exchange
    else:
        return '1'  # Default to Shanghai


def get_stock_data_eastmoney_all_history(stock_code="002354"):
    """
    Fetch all historical stock data using East Money API
    """
    try:
        print(f"Fetching all historical data for stock {stock_code} from East Money...")

        # Get market type
        market = get_stock_market(stock_code)
        secid = f"{market}.{stock_code}"

        # Use East Money API to fetch all historical data
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"

        # Set an early enough start date (Chinese stock market started in 1990)
        start_date = "19900101"
        end_date = datetime.now().strftime('%Y%m%d')

        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # Daily candlestick
            'fqt': '1',  # Forward-adjusted
            'beg': start_date,
            'end': end_date,
            'lmt': '50000',  # Increased limit for more historical data
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'cb': f'jQuery{random.randint(1000000, 9999999)}_{int(time.time() * 1000)}'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/',
            'Accept': '*/*',
        }

        time.sleep(random.uniform(1, 2))

        response = requests.get(url, params=params, headers=headers, timeout=15)

        print(f"API response status: {response.status_code}")

        if response.status_code == 200:
            # Handle JSONP response
            response_text = response.text

            # Extract JSON data (handle JSONP format)
            if response_text.startswith('/**/'):
                response_text = response_text[4:]

            # Find JSON data start and end positions
            start_idx = response_text.find('(')
            end_idx = response_text.rfind(')')

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx + 1:end_idx]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    print("❌ JSON parse failed, trying direct parse...")
                    return parse_kline_data_directly_all_history(response_text, stock_code)
            else:
                print("❌ Cannot find JSON data boundaries")
                return None

            print(f"API data status: {data.get('rc', 'N/A')}")

            if data and data.get('data') is not None:
                klines = data['data'].get('klines', [])
                print(f"Retrieved {len(klines)} historical candlestick records")

                if not klines:
                    print("⚠️ Candlestick data is empty")
                    return None

                # Parse data
                stock_data = []
                for kline in klines:
                    try:
                        items = kline.split(',')
                        if len(items) >= 6:
                            stock_data.append({
                                'date': items[0],
                                'stock_code': stock_code,
                                'open': float(items[1]),
                                'close': float(items[2]),
                                'high': float(items[3]),
                                'low': float(items[4]),
                                'volume': float(items[5]),
                                'amount': float(items[6]) if len(items) > 6 else 0,
                                'amplitude': float(items[7]) if len(items) > 7 else 0,
                                'pct_change': float(items[8]) if len(items) > 8 else 0,
                                'price_change': float(items[9]) if len(items) > 9 else 0,
                                'turnover': float(items[10]) if len(items) > 10 else 0
                            })
                    except (ValueError, IndexError) as e:
                        continue

                if not stock_data:
                    print("❌ No valid data after parsing")
                    return None

                df = pd.DataFrame(stock_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()

                print(f"✅ Successfully retrieved {len(df)} historical records")
                print(
                    f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
                return df
            else:
                print("❌ API returned empty data")
                return None
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error fetching historical data: {str(e)}")
        return None


def parse_kline_data_directly_all_history(response_text, stock_code):
    """
    Parse candlestick data directly (used when JSON parsing fails) - full history version
    """
    try:
        # Try to extract candlestick data directly from response text
        if '"klines":[' in response_text:
            start_idx = response_text.find('"klines":[') + 10
            end_idx = response_text.find(']', start_idx)
            klines_str = response_text[start_idx:end_idx]

            # Clean and split
            klines = [k.strip().strip('"') for k in klines_str.split('","') if k.strip()]

            stock_data = []
            for kline in klines:
                if kline.strip():
                    items = kline.split(',')
                    if len(items) >= 6:
                        stock_data.append({
                            'date': items[0],
                            'stock_code': stock_code,
                            'open': float(items[1]),
                            'close': float(items[2]),
                            'high': float(items[3]),
                            'low': float(items[4]),
                            'volume': float(items[5]),
                            'amount': float(items[6]) if len(items) > 6 else 0,
                        })

            if stock_data:
                df = pd.DataFrame(stock_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                print(f"✅ Direct parse retrieved {len(df)} historical records")
                return df
    except Exception as e:
        print(f"❌ Direct parse also failed: {e}")

    return None


def get_stock_data_akshare_all_history(stock_code="002354"):
    """
    Use AKShare as a backup data source - full history version
    """
    try:
        print(f"Trying AKShare to fetch all historical data for stock {stock_code}...")
        import akshare as ak

        # Get all historical data
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                adjust="qfq")

        if df is not None and not df.empty:
            # Rename columns to match our format
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'price_change',
                '换手率': 'turnover'
            }

            # Only map existing columns
            actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=actual_mapping)

            # Add stock code column
            df['stock_code'] = stock_code
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()

            print(f"✅ AKShare successfully retrieved {len(df)} historical records")
            print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            return df
        else:
            print("❌ AKShare returned no data")
            return None

    except ImportError:
        print("⚠️ AKShare not installed, use: pip install akshare")
        return None
    except Exception as e:
        print(f"❌ AKShare historical data fetch failed: {e}")
        return None


def get_stock_data_baostock_all_history(stock_code="002354"):
    """
    Use Baostock as a third data source - full history version
    """
    try:
        print(f"Trying Baostock to fetch all historical data for stock {stock_code}...")
        import baostock as bs
        import pandas as pd

        # Login
        lg = bs.login()

        # Add market prefix based on exchange
        market = get_stock_market(stock_code)
        if market == '0':
            full_code = f"sz.{stock_code}"
        else:
            full_code = f"sh.{stock_code}"

        # Get listing date
        rs = bs.query_stock_basic(code=full_code)
        if rs.error_code != '0':
            print(f"❌ Failed to get stock basic info: {rs.error_msg}")
            bs.logout()
            return None

        # Get listing date
        list_date = None
        while (rs.error_code == '0') & rs.next():
            list_date = rs.get_row_data()[2]  # listing date is the third field

        if not list_date:
            print("❌ Cannot get listing date")
            bs.logout()
            return None

        print(f"Stock listing date: {list_date}")

        # Fetch all data from listing date to now
        end_date = datetime.now().strftime('%Y-%m-%d')

        rs = bs.query_history_k_data_plus(
            full_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=list_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # Forward-adjusted
        )

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        # Logout
        bs.logout()

        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)

            # Convert data types
            df['date'] = pd.to_datetime(df['date'])
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
            df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')

            # Rename columns
            df = df.rename(columns={
                'turn': 'turnover',
                'pctChg': 'pct_change'
            })

            # Add stock code column
            df['stock_code'] = stock_code
            df.set_index('date', inplace=True)
            df = df.sort_index()

            # Calculate price change amount
            df['price_change'] = df['close'].diff()

            # Remove invalid data
            df = df.dropna()

            print(f"✅ Baostock successfully retrieved {len(df)} historical records")
            print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            return df
        else:
            print("❌ Baostock returned no data")
            return None

    except ImportError:
        print("⚠️ Baostock not installed, use: pip install baostock")
        return None
    except Exception as e:
        print(f"❌ Baostock historical data fetch failed: {e}")
        return None


def get_stock_data_with_retry_all_history(stock_code="002354", retry_count=2):
    """
    Data fetch with retry - multi-source full history version
    """
    data_sources = [
        ("AKShare", get_stock_data_akshare_all_history),
        ("Baostock", get_stock_data_baostock_all_history),
        ("East Money", get_stock_data_eastmoney_all_history)
    ]

    for source_name, data_func in data_sources:
        print(f"\n🔍 Trying to fetch all historical data from {source_name}...")
        data = data_func(stock_code)

        if data is not None and not data.empty:
            print(f"✅ {source_name} historical data fetch successful!")
            # Mark data source
            data.attrs['data_source'] = source_name
            return data

    print("❌ All real data sources failed, using sample data...")
    return create_sample_data_all_history(stock_code)


def create_sample_data_all_history(stock_code="002354"):
    """
    Create more realistic historical sample data - starting from listing year
    """
    # Simulated listing years for different stocks
    list_years = {
        '600580': 2002,  # Wolong Electric Drive
        '002354': 2010,  # Tianyu Digital Technology
        '300418': 2015,  # Kunlun Wanwei
        '300207': 2011,  # Sunwoda
    }

    list_year = list_years.get(stock_code, 2010)
    current_year = datetime.now().year

    print(f"📊 Creating sample data for {stock_code} from listing year {list_year} to present...")

    # Generate trading days from listing year to now (excluding weekends)
    start_date = datetime(list_year, 1, 1)
    end_date = datetime.now()
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')

    # Generate more realistic price data
    import numpy as np
    np.random.seed(42)

    # Set reasonable base price (by stock type)
    base_prices = {
        '600580': 8.0,   # Wolong Electric Drive
        '002354': 15.0,  # Tianyu Digital Technology - higher IPO price
        '300418': 20.0,  # Kunlun Wanwei
        '300207': 12.0,  # Sunwoda
    }
    base_price = base_prices.get(stock_code, 10.0)

    stock_data = []
    current_price = base_price

    for i, date in enumerate(all_dates):
        # Simulate realistic market volatility
        volatility = 0.02  # 2% daily volatility

        if i > 0:
            # Use random walk to simulate price changes
            daily_return = np.random.normal(0, volatility)

            # Simulate different market trends by year
            year = date.year
            if year <= list_year + 2:  # Higher volatility in early post-IPO years
                daily_return += np.random.normal(0.001, 0.01)
            elif year <= list_year + 5:  # Growth phase
                daily_return += np.random.normal(0.0005, 0.005)
            else:  # Mature phase
                daily_return += np.random.normal(0.0002, 0.003)

            current_price = current_price * (1 + daily_return)

            # Price boundary limits
            current_price = max(base_price * 0.3, min(base_price * 10.0, current_price))
        else:
            current_price = base_price

        # Generate OHLC data
        open_variation = np.random.normal(0, volatility * 0.2)
        open_price = current_price * (1 + open_variation)

        daily_range = abs(np.random.normal(volatility * 0.8, volatility * 0.3))
        high_price = max(open_price, current_price) * (1 + daily_range)
        low_price = min(open_price, current_price) * (1 - daily_range)
        close_price = current_price

        # Ensure price validity
        high_price = max(open_price, close_price, low_price, high_price)
        low_price = min(open_price, close_price, high_price, low_price)

        # Generate volume (growing over years)
        year = date.year
        base_volume = 100000 + (year - list_year) * 50000  # Volume grows year by year
        volume_variation = abs(daily_return) * 5000000 if i > 0 else 0
        volume = int(base_volume + volume_variation + np.random.randint(-200000, 400000))
        volume = max(50000, volume)

        # Calculate amount (in 10k CNY)
        amount = volume * close_price / 10000

        # Calculate price change and percentage
        if i > 0:
            prev_close = stock_data[-1]['close']
            price_change = close_price - prev_close
            pct_change = (price_change / prev_close) * 100
        else:
            price_change = 0
            pct_change = 0

        # Calculate amplitude
        amplitude = ((high_price - low_price) / open_price) * 100

        # Generate turnover rate (between 1%-15%)
        turnover_rate = np.random.uniform(1.0, 15.0)

        stock_data.append({
            'date': date,
            'stock_code': stock_code,
            'open': round(open_price, 2),
            'close': round(close_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'volume': volume,
            'amount': round(amount, 2),
            'amplitude': round(amplitude, 2),
            'pct_change': round(pct_change, 2),
            'price_change': round(price_change, 2),
            'turnover': round(turnover_rate, 2)
        })

    df = pd.DataFrame(stock_data)
    df.set_index('date', inplace=True)

    print(f"✅ Created {len(df)} simulated historical records from {list_year} to present")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Mark as simulated data
    df.attrs['data_source'] = 'simulated_historical_data'

    return df


def display_all_history_data_info(df, stock_code):
    """Display full historical data summary"""
    if df is None or df.empty:
        print("No data to display")
        return

    # Get data source
    data_source = df.attrs.get('data_source', 'unknown source')

    print(f"\n{'=' * 60}")
    print(f"Stock {stock_code} Full Historical Data Summary")
    print(f"{'=' * 60}")

    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(df):,}")
    print(f"Data source: {data_source}")

    # Display statistics by year
    years = sorted(df.index.year.unique())
    print(f"\nHistorical years: {years}")

    # Show statistics for key years
    key_years = [years[0]]  # listing year
    if len(years) > 1:
        key_years.append(years[-1])  # latest year
    if len(years) > 5:
        key_years.extend([years[len(years) // 2], years[len(years) // 4], years[3 * len(years) // 4]])

    close_col = 'close' if 'close' in df.columns else ('收盘价' if '收盘价' in df.columns else df.columns[0])
    high_col = 'high' if 'high' in df.columns else ('最高价' if '最高价' in df.columns else None)
    low_col = 'low' if 'low' in df.columns else ('最低价' if '最低价' in df.columns else None)

    for year in sorted(set(key_years)):
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            print(f"\n{year} Statistics:")
            print(f"  Trading days: {len(year_data)}")
            print(f"  Avg close: {year_data[close_col].mean():.2f}")
            if high_col:
                print(f"  High: {year_data[high_col].max():.2f}")
            if low_col:
                print(f"  Low: {year_data[low_col].min():.2f}")
            if len(year_data) > 1:
                year_return = (year_data[close_col].iloc[-1] / year_data[close_col].iloc[0] - 1) * 100
                print(f"  Annual return: {year_return:+.2f}%")

    # Overall statistics
    print(f"\nOverall Statistics:")
    total_return = (df[close_col].iloc[-1] / df[close_col].iloc[0] - 1) * 100
    print(f"  Total return: {total_return:+.2f}%")
    if high_col:
        print(f"  Historical high: {df[high_col].max():.2f}")
    if low_col:
        print(f"  Historical low: {df[low_col].min():.2f}")
    volume_col = 'volume' if 'volume' in df.columns else ('成交量' if '成交量' in df.columns else None)
    if volume_col:
        print(f"  Avg daily volume: {df[volume_col].mean():,.0f} shares")

    # Show latest trading day data
    latest_date = df.index.max()
    print(f"\nLatest trading day ({latest_date.strftime('%Y-%m-%d')}) data:")
    latest_data = df.loc[latest_date]
    for col, value in latest_data.items():
        if col != 'stock_code':
            if col in ['volume']:
                print(f"  {col}: {value:,.0f}")
            elif col in ['amount']:
                print(f"  {col}: {value:,.2f} (10k CNY)")
            else:
                print(f"  {col}: {value}")


def save_all_history_stock_data(df, stock_code, save_dir="D:/lianghuajiaoyi/Kronos/examples/data"):
    """
    Save full historical stock data to the specified directory
    """
    if df is not None and not df.empty:
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save CSV file - use full-history naming
        csv_file = os.path.join(save_dir, f"{stock_code}_all_history.csv")

        # Reset index to save date column
        df_reset = df.reset_index()
        df_reset.to_csv(csv_file, encoding='utf-8-sig', index=False)

        print(f"\n📁 Full historical data saved: {csv_file}")

        # Also save a version split by year
        date_col = 'date' if 'date' in df_reset.columns else df_reset.columns[0]
        years = df_reset[date_col].dt.year.unique()
        for year in years:
            year_data = df_reset[df_reset[date_col].dt.year == year]
            year_file = os.path.join(save_dir, f"{stock_code}_{year}.csv")
            year_data.to_csv(year_file, encoding='utf-8-sig', index=False)

        print(f"📁 Also saved {len(years)} individual yearly data files")
        return True
    return False


def main_all_history(stock_code="002354"):
    """
    Main function: fetch and save full historical stock data
    """
    # Set save directory
    save_directory = "D:/lianghuajiaoyi/Kronos/examples/data"

    print("=" * 60)
    print(f"Fetching all historical data for stock {stock_code}")
    print("=" * 60)
    print(f"Data will be saved to: {save_directory}")

    # Check required libraries
    try:
        import requests
        import numpy as np
    except ImportError:
        print("Installing required libraries...")
        import subprocess
        subprocess.check_call(["pip", "install", "requests", "numpy", "pandas"])
        import requests
        import numpy as np

    # Fetch full historical data (multi-source)
    stock_data = get_stock_data_with_retry_all_history(stock_code)

    if stock_data is not None:
        # Display data info
        display_all_history_data_info(stock_data, stock_code)

        # Save full historical data to the specified directory
        save_all_history_stock_data(stock_data, stock_code, save_directory)

        print(f"\n🎉 Full historical data processing complete for stock {stock_code}!")
        print(
            f"Date range: {stock_data.index.min().strftime('%Y-%m-%d')} to {stock_data.index.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(stock_data):,}")

        # Show saved file info
        csv_file = os.path.join(save_directory, f"{stock_code}_all_history.csv")
        if os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file) / 1024  # KB
            print(f"📄 Generated file: {csv_file} ({file_size:.1f} KB)")
    else:
        print("❌ Failed to retrieve full historical data")


# Usage instructions
if __name__ == "__main__":
    """
    Usage:
    Modify the parameters below to fetch full historical data for different stocks
    """

    # ==================== Modify parameters here ====================
    TARGET_STOCK_CODE = "300418"  # Stock code
    # ===============================================================

    print("Stock Full History Data Fetcher")
    print("Note: Modify TARGET_STOCK_CODE in the code to fetch data for different stocks")
    print(f"Current setting: stock code={TARGET_STOCK_CODE}")
    print()

    # Run main program
    main_all_history(stock_code=TARGET_STOCK_CODE)

    print(f"\n💡 Tip: To fetch full historical data for other stocks, modify TARGET_STOCK_CODE in the code")
