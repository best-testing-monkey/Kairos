# run_backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class KronosBacktester:
    """Kronos model backtester"""

    def __init__(self, data_dir, model_dir, initial_capital=100000):
        """
        Initialize backtester

        Parameters:
        data_dir: data directory
        model_dir: model prediction results directory
        initial_capital: initial capital
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.results = {}

    def load_historical_data(self, stock_code):
        """Load historical data"""
        csv_file = os.path.join(self.data_dir, f"{stock_code}_stock_data.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Data file not found: {csv_file}")

        df = pd.read_csv(csv_file, encoding='utf-8-sig')

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

    def load_predictions(self, stock_code):
        """Load model prediction results"""
        pred_files = [
            os.path.join(self.model_dir, f"{stock_code}_kronos_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_detailed_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_predictions.csv")
        ]

        pred_df = None
        for pred_file in pred_files:
            if os.path.exists(pred_file):
                pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
                print(f"✅ Found prediction file: {pred_file}")
                break

        if pred_df is None:
            raise FileNotFoundError(f"No prediction file found, check directory: {self.model_dir}")

        column_mapping = {
            '日期': 'date',
            '预测收盘价': 'predicted_close',
            '收盘价': 'predicted_close',
            '预测成交量': 'predicted_volume',
            '成交量': 'predicted_volume'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in pred_df.columns:
                pred_df = pred_df.rename(columns={old_col: new_col})

        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df.set_index('date', inplace=True)
        pred_df = pred_df.sort_index()

        print(f"✅ Loaded prediction data: {len(pred_df)} records")
        print(f"Prediction range: {pred_df.index.min()} to {pred_df.index.max()}")

        return pred_df

    def align_data(self, hist_df, pred_df):
        """Align historical data and prediction data time ranges"""
        last_hist_date = hist_df.index.max()

        pred_df_aligned = pred_df[pred_df.index > last_hist_date]

        if len(pred_df_aligned) == 0:
            pred_df_aligned = pred_df.copy()
            print("⚠️ Warning: No future dates in prediction data, using all prediction data")

        print(f"✅ Data aligned: history ends at {last_hist_date}, predictions start at {pred_df_aligned.index.min()}")

        return pred_df_aligned

    def calculate_trading_signals(self, hist_df, pred_df, threshold=0.02):
        """Calculate trading signals"""
        pred_df = self.align_data(hist_df, pred_df)

        combined = pd.concat([
            hist_df[['close']].rename(columns={'close': 'actual'}),
            pred_df[['predicted_close']].rename(columns={'predicted_close': 'predicted'})
        ], axis=1)

        combined['pred_return'] = combined['predicted'].pct_change()

        combined['signal'] = 0
        combined['signal'] = np.where(combined['pred_return'] > threshold, 1,
                                      np.where(combined['pred_return'] < -threshold, -1, 0))

        combined['position'] = combined['signal'].replace(to_replace=0, method='ffill').fillna(0)

        return combined

    def run_backtest(self, combined_df):
        """Run backtest"""
        capital = self.initial_capital
        position = 0
        trades = []

        backtest_results = pd.DataFrame(index=combined_df.index)
        backtest_results['capital'] = capital
        backtest_results['position'] = 0
        backtest_results['returns'] = 0.0
        backtest_results['price'] = combined_df['actual'].combine_first(combined_df['predicted'])

        for i, (date, row) in enumerate(combined_df.iterrows()):
            current_price = row['actual'] if not pd.isna(row['actual']) else row['predicted']
            signal = row['position']

            if pd.isna(current_price):
                continue

            if i > 0:
                prev_position = backtest_results['position'].iloc[i - 1] if i > 0 else 0

                if prev_position != 0 and signal == 0:
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': prev_position,
                        'capital': capital
                    })

                elif prev_position == 0 and signal != 0:
                    shares = int(capital / current_price)
                    if shares > 0:
                        position = shares * signal
                        capital -= shares * current_price
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares * signal,
                            'capital': capital
                        })

            portfolio_value = capital + position * current_price

            backtest_results.loc[date, 'capital'] = portfolio_value
            backtest_results.loc[date, 'position'] = position
            backtest_results.loc[date, 'price'] = current_price

            if i > 0:
                prev_value = backtest_results['capital'].iloc[i - 1]
                if prev_value > 0:
                    backtest_results.loc[date, 'returns'] = (portfolio_value - prev_value) / prev_value

        return backtest_results, trades

    def calculate_metrics(self, backtest_results, trades):
        """Calculate backtest metrics"""
        returns = backtest_results['returns'].replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_trade_return': 0,
                'trade_count': 0,
                'final_capital': self.initial_capital
            }

        total_return = (backtest_results['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        volatility = returns.std() * np.sqrt(252)

        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        trade_returns = []
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]
            trade_return = (sell['price'] - buy['price']) / buy['price']
            trade_returns.append(trade_return)

        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'trade_count': len(trades),
            'final_capital': backtest_results['capital'].iloc[-1]
        }

        return metrics

    def plot_backtest_results(self, backtest_results, metrics, stock_code, output_dir):
        """Plot backtest result charts"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        ax1.plot(backtest_results.index, backtest_results['capital'],
                 linewidth=2, label='Strategy Equity Curve', color='#1f77b4')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--',
                    label=f'Initial Capital ({self.initial_capital:,.0f} CNY)')
        ax1.set_ylabel('Capital (CNY)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{stock_code} Kronos Model Backtest Results', fontsize=14, fontweight='bold')

        cumulative_returns = (1 + backtest_results['returns'].fillna(0)).cumprod()
        ax2.plot(backtest_results.index, cumulative_returns,
                 linewidth=2, label='Strategy Cumulative Return', color='#2ca02c')

        price_returns = backtest_results['price'].pct_change().fillna(0)
        benchmark_returns = (1 + price_returns).cumprod()
        ax2.plot(backtest_results.index, benchmark_returns,
                 linewidth=2, label='Benchmark (Buy-and-Hold)', color='#ff7f0e', alpha=0.7)

        ax2.set_ylabel('Cumulative Return', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        ax3.fill_between(backtest_results.index, drawdown, 0,
                         alpha=0.3, color='red', label='Drawdown')
        ax3.set_ylabel('Drawdown', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annual_return']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Trade Count: {metrics['trade_count']}\n"
            f"Final Capital: {metrics['final_capital']:,.0f} CNY"
        )

        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        chart_file = os.path.join(output_dir, f'{stock_code}_backtest_results.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Backtest chart saved: {chart_file}")

        plt.show()

    def run_complete_backtest(self, stock_code, output_dir, threshold=0.02):
        """Run complete backtest workflow"""
        print(f"🎯 Starting {stock_code} backtest analysis")
        print("=" * 50)

        try:
            print("Step 1: Loading historical data and predictions...")
            hist_df = self.load_historical_data(stock_code)
            pred_df = self.load_predictions(stock_code)

            print("Step 2: Calculating trading signals...")
            combined_df = self.calculate_trading_signals(hist_df, pred_df, threshold)

            print("Step 3: Running backtest...")
            backtest_results, trades = self.run_backtest(combined_df)

            print("Step 4: Calculating backtest metrics...")
            metrics = self.calculate_metrics(backtest_results, trades)

            print("Step 5: Generating backtest charts...")
            self.plot_backtest_results(backtest_results, metrics, stock_code, output_dir)

            print("\n" + "=" * 70)
            print(f"📊 {stock_code} Backtest Report")
            print("=" * 70)
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'return' in key or 'drawdown' in key or 'rate' in key:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            print(f"\nTrade history (last {min(10, len(trades))} trades):")
            for i, trade in enumerate(trades[-10:], 1):
                print(f"  Trade {i}: {trade['date'].strftime('%Y-%m-%d')} "
                      f"{trade['action']} {abs(trade['shares'])} shares @ {trade['price']:.2f} CNY")

            return metrics, backtest_results, trades

        except Exception as e:
            print(f"❌ Error during backtest: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """Main function: run Kronos model backtest"""
    BACKTEST_CONFIG = {
        "stock_code": "000831",
        "data_dir": r"D:\lianghuajiaoyi\Kronos\examples\data",
        "model_dir": r"D:\lianghuajiaoyi\Kronos\examples\yuce",
        "output_dir": r"D:\lianghuajiaoyi\Kronos\examples\backtest",
        "initial_capital": 100000,
        "threshold": 0.02
    }

    print("🤖 Kronos Model Backtest System")
    print("=" * 50)
    print(f"Backtest stock: {BACKTEST_CONFIG['stock_code']}")
    print(f"Initial capital: {BACKTEST_CONFIG['initial_capital']:,.0f} CNY")
    print(f"Trading threshold: {BACKTEST_CONFIG['threshold']:.1%}")
    print()

    backtester = KronosBacktester(
        data_dir=BACKTEST_CONFIG["data_dir"],
        model_dir=BACKTEST_CONFIG["model_dir"],
        initial_capital=BACKTEST_CONFIG["initial_capital"]
    )

    metrics, results, trades = backtester.run_complete_backtest(
        stock_code=BACKTEST_CONFIG["stock_code"],
        output_dir=BACKTEST_CONFIG["output_dir"],
        threshold=BACKTEST_CONFIG["threshold"]
    )

    if metrics:
        print(f"\n✅ {BACKTEST_CONFIG['stock_code']} backtest complete!")
        print(f"📁 Results saved to: {BACKTEST_CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
