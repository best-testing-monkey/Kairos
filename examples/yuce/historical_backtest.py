# historical_backtest.py
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class HistoricalBacktester:
    """
    Historical backtester: validates model prediction accuracy using historical data
    """

    def __init__(self, data_dir, initial_capital=100000):
        self.data_dir = data_dir
        self.initial_capital = initial_capital

    def load_historical_data(self, stock_code):
        """Load historical data"""
        csv_file = os.path.join(self.data_dir, f"{stock_code}_stock_data.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Data file not found: {csv_file}")

        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        # Standardize column names
        column_mapping = {
            '日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '成交额': 'amount'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()

        print(f"✅ Loaded historical data: {len(df)} records")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def simulate_model_prediction(self, df, lookback_days=60, pred_days=30):
        """
        Simulate model predictions: use historical data to 'predict', then compare with actual results
        """
        results = []

        # Select multiple time points from data for 'prediction'
        test_points = range(lookback_days, len(df) - pred_days, pred_days)

        for start_idx in test_points:
            # Simulate prediction: use previous lookback_days data to 'predict' next pred_days
            historical_data = df.iloc[start_idx - lookback_days:start_idx]
            actual_future = df.iloc[start_idx:start_idx + pred_days]

            # Simple prediction strategy (replace with your actual model prediction)
            # Using moving average as example prediction
            pred_close = self.simple_prediction(historical_data, pred_days)

            # Record results
            for i in range(min(len(pred_close), len(actual_future))):
                results.append({
                    'date': actual_future.index[i],
                    'actual_close': actual_future['close'].iloc[i],
                    'predicted_close': pred_close[i],
                    'lookback_start': historical_data.index[0],
                    'prediction_date': historical_data.index[-1]
                })

        return pd.DataFrame(results)

    def simple_prediction(self, historical_data, pred_days):
        """Simple prediction method (example)"""
        # Use moving average + random noise as prediction
        last_price = historical_data['close'].iloc[-1]
        avg_volatility = historical_data['close'].pct_change().std()

        predictions = []
        current_price = last_price

        for _ in range(pred_days):
            # Simulate price change (normal distribution)
            change = np.random.normal(0, avg_volatility)
            current_price = current_price * (1 + change)
            predictions.append(current_price)

        return predictions

    def calculate_prediction_accuracy(self, results_df):
        """Calculate prediction accuracy metrics"""
        results_df['error'] = results_df['predicted_close'] - results_df['actual_close']
        results_df['error_pct'] = results_df['error'] / results_df['actual_close']
        results_df['abs_error_pct'] = abs(results_df['error_pct'])

        accuracy_metrics = {
            'mean_absolute_error_rate': results_df['abs_error_pct'].mean(),
            'prediction_accuracy': (results_df['abs_error_pct'] < 0.05).mean(),  # error < 5% counts as accurate
            'direction_accuracy': (np.sign(results_df['predicted_close'].diff()) ==
                           np.sign(results_df['actual_close'].diff())).mean(),
            'correlation': results_df['predicted_close'].corr(results_df['actual_close'])
        }

        return accuracy_metrics

    def run_trading_strategy(self, results_df, threshold=0.03):
        """Run trading strategy based on prediction results"""
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []

        # Sort by date
        results_df = results_df.sort_index()

        for date, row in results_df.iterrows():
            current_price = row['actual_close']
            predicted_price = row['predicted_close']
            predicted_return = (predicted_price - current_price) / current_price

            # Trading logic
            if position == 0 and predicted_return > threshold:
                # Buy signal
                shares = int(capital / current_price)
                if shares > 0:
                    position = shares
                    capital -= shares * current_price
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'reason': f'Predicted rise {predicted_return:.2%}'
                    })

            elif position > 0 and predicted_return < -threshold:
                # Sell signal
                capital += position * current_price
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'reason': f'Predicted fall {predicted_return:.2%}'
                })
                position = 0

            # Calculate current total portfolio value
            portfolio_value = capital + position * current_price
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': position,
                'price': current_price
            })

        return pd.DataFrame(portfolio_values), trades

    def calculate_performance(self, portfolio_df, trades):
        """Calculate strategy performance metrics"""
        portfolio_df = portfolio_df.set_index('date')
        returns = portfolio_df['portfolio_value'].pct_change().dropna()

        total_return = (portfolio_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital

        if len(returns) > 0:
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            annual_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # Buy-and-hold strategy comparison
        buy_hold_return = (portfolio_df['price'].iloc[-1] - portfolio_df['price'].iloc[0]) / portfolio_df['price'].iloc[
            0]

        performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'buy_hold_return': buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': len(trades),
            'final_capital': portfolio_df['portfolio_value'].iloc[-1],
            'excess_return': total_return - buy_hold_return
        }

        return performance

    def plot_comparison(self, results_df, portfolio_df, stock_code, output_dir):
        """Plot prediction comparison charts"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # 1. Price prediction comparison
        ax1.plot(results_df.index, results_df['actual_close'],
                 label='Actual Price', color='blue', linewidth=2)
        ax1.plot(results_df.index, results_df['predicted_close'],
                 label='Predicted Price', color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Price (CNY)')
        ax1.legend()
        ax1.set_title(f'{stock_code} - Predicted vs Actual Price', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Prediction error
        ax2.bar(results_df.index, results_df['error_pct'] * 100,
                alpha=0.6, color='orange')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Prediction Error (%)')
        ax2.set_title('Prediction Error Analysis')
        ax2.grid(True, alpha=0.3)

        # 3. Strategy performance
        ax3.plot(portfolio_df['date'], portfolio_df['portfolio_value'],
                 label='Strategy Equity Curve', color='green', linewidth=2)
        ax3.axhline(y=self.initial_capital, color='red', linestyle='--',
                    label=f'Initial Capital ({self.initial_capital:,.0f})')

        # Buy-and-hold comparison
        initial_shares = self.initial_capital / portfolio_df['price'].iloc[0]
        buy_hold_values = portfolio_df['price'] * initial_shares
        ax3.plot(portfolio_df['date'], buy_hold_values,
                 label='Buy-and-Hold Strategy', color='blue', linestyle=':', alpha=0.7)

        ax3.set_ylabel('Capital (CNY)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.set_title('Strategy Performance Comparison')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f'{stock_code}_historical_backtest.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Historical backtest chart saved: {chart_file}")

        plt.close('all')

    def run_complete_backtest(self, stock_code, output_dir, lookback_days=60, pred_days=30, threshold=0.03):
        """Run complete historical backtest"""
        print(f"🎯 Starting {stock_code} historical backtest analysis")
        print("=" * 60)

        try:
            # 1. Load historical data
            print("Step 1: Loading historical data...")
            df = self.load_historical_data(stock_code)

            # 2. Simulate model predictions
            print("Step 2: Simulating model predictions...")
            results_df = self.simulate_model_prediction(df, lookback_days, pred_days)

            # 3. Calculate prediction accuracy
            print("Step 3: Calculating prediction accuracy...")
            accuracy_metrics = self.calculate_prediction_accuracy(results_df)

            # 4. Run trading strategy
            print("Step 4: Running trading strategy...")
            portfolio_df, trades = self.run_trading_strategy(results_df, threshold)

            # 5. Calculate strategy performance
            print("Step 5: Calculating strategy performance...")
            performance = self.calculate_performance(portfolio_df, trades)

            # 6. Plot results
            print("Step 6: Generating backtest charts...")
            self.plot_comparison(results_df, portfolio_df, stock_code, output_dir)

            # 7. Print report
            print("\n" + "=" * 70)
            print(f"📊 {stock_code} Historical Backtest Report")
            print("=" * 70)

            print("\n🔍 Prediction Accuracy Analysis:")
            for metric, value in accuracy_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.4f}")

            print("\n💰 Strategy Performance Analysis:")
            for metric, value in performance.items():
                if isinstance(value, float):
                    if 'return' in metric or 'drawdown' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

            print(f"\n📈 Trading Statistics:")
            print(f"  Total trades: {len(trades)}")
            print(f"  Buy trades: {len([t for t in trades if t['action'] == 'BUY'])}")
            print(f"  Sell trades: {len([t for t in trades if t['action'] == 'SELL'])}")

            if len(trades) > 0:
                print(f"\nLast 5 trades:")
                for trade in trades[-5:]:
                    print(f"  {trade['date'].strftime('%Y-%m-%d')} {trade['action']} "
                          f"{trade['shares']} shares @ {trade['price']:.2f} - {trade['reason']}")

            return accuracy_metrics, performance, results_df

        except Exception as e:
            print(f"❌ Error during backtest: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """Main function"""
    # Configuration parameters
    BACKTEST_CONFIG = {
        "stock_code": "300418",
        "data_dir": r"./examples/data",
        "output_dir": r"./examples/historical_backtest",
        "initial_capital": 100000,
        "lookback_days": 60,  # use 60 days of historical data
        "pred_days": 30,  # predict 30 days
        "threshold": 0.03  # 3% trading threshold
    }

    print("🤖 Kronos Model Historical Backtest System")
    print("=" * 50)
    print(f"Backtest stock: {BACKTEST_CONFIG['stock_code']}")
    print(f"Lookback days: {BACKTEST_CONFIG['lookback_days']} days")
    print(f"Prediction days: {BACKTEST_CONFIG['pred_days']} days")
    print(f"Initial capital: {BACKTEST_CONFIG['initial_capital']:,.0f}")
    print()

    # Create backtester and run
    backtester = HistoricalBacktester(
        data_dir=BACKTEST_CONFIG["data_dir"],
        initial_capital=BACKTEST_CONFIG["initial_capital"]
    )

    accuracy, performance, results = backtester.run_complete_backtest(
        stock_code=BACKTEST_CONFIG["stock_code"],
        output_dir=BACKTEST_CONFIG["output_dir"],
        lookback_days=BACKTEST_CONFIG["lookback_days"],
        pred_days=BACKTEST_CONFIG["pred_days"],
        threshold=BACKTEST_CONFIG["threshold"]
    )

    if accuracy and performance:
        print(f"\n✅ {BACKTEST_CONFIG['stock_code']} historical backtest complete!")

        # Simple conclusion
        if performance['excess_return'] > 0:
            print("🎉 Conclusion: Strategy outperformed buy-and-hold!")
        else:
            print("⚠️ Conclusion: Strategy did not outperform buy-and-hold.")

        print(f"📁 Detailed results saved to: {BACKTEST_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
