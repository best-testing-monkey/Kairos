"""Tests for pipeline automation helpers."""

import pytest
import tempfile
import sqlite3
import os
import pandas as pd
from kairos_strategies import _period_to_weeks
from kairos_pipeline import build_viability_report, get_connection, SCHEMA, insert_oracle_row, insert_model_row


class TestPeriodToWeeks:
    """Test the _period_to_weeks period parsing helper."""

    def test_period_to_weeks_values(self):
        """Test period-to-weeks conversion with standard values."""
        # 6m: 6 * (365.25/12) / 7 ≈ 26.089
        assert abs(_period_to_weeks("6m") - 26.09) < 1e-2

        # 1m: 1 * (365.25/12) / 7 ≈ 4.348
        assert abs(_period_to_weeks("1m") - 4.35) < 1e-2

        # 2w: 2 weeks exactly
        assert abs(_period_to_weeks("2w") - 2.0) < 1e-2

        # 1y: 365.25 / 7 ≈ 52.179
        assert abs(_period_to_weeks("1y") - 52.18) < 1e-2

    def test_period_to_weeks_single_unit(self):
        """Test single-unit periods."""
        # 1d: 1 / 7 ≈ 0.143
        assert abs(_period_to_weeks("1d") - 1.0/7.0) < 1e-6

        # 1w: 1 week exactly
        assert _period_to_weeks("1w") == 1.0

        # 1m: 365.25/12/7 ≈ 4.348
        assert abs(_period_to_weeks("1m") - 365.25/12/7) < 1e-6

    def test_period_to_weeks_invalid(self):
        """Test that invalid period strings raise ValueError."""
        # Invalid formats should raise ValueError with the same error type as _period_to_bars
        with pytest.raises(ValueError) as exc_info:
            _period_to_weeks("invalid")
        assert "Unrecognised backtest_period" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            _period_to_weeks("6x")
        assert "Unrecognised backtest_period" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            _period_to_weeks("m")
        assert "Unrecognised backtest_period" in str(exc_info.value)

    def test_period_to_weeks_case_insensitive(self):
        """Test that period strings are case-insensitive."""
        assert _period_to_weeks("6M") == _period_to_weeks("6m")
        assert _period_to_weeks("1Y") == _period_to_weeks("1y")
        assert _period_to_weeks("2W") == _period_to_weeks("2w")

    def test_period_to_weeks_whitespace_tolerant(self):
        """Test that leading/trailing whitespace is handled."""
        assert _period_to_weeks(" 6m ") == _period_to_weeks("6m")
        assert _period_to_weeks("\t1y\t") == _period_to_weeks("1y")


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with schema for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()

    yield conn

    conn.close()
    try:
        os.remove(db_path)
    except OSError:
        pass


class TestViabilityReport:
    """Tests for build_viability_report function."""

    def test_report_columns_exact(self, temp_db):
        """Verify report columns are exactly in order as specified."""
        # Insert one oracle and one base row
        oracle_row = {
            "strategy_name": "test_strat",
            "sharpe": 1.5,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "BTC-USD,ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, oracle_row)

        base_row = {
            "stage": "base",
            "strategy_name": "test_strat",
            "sharpe": 1.2,
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "BTC-USD,ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": "/path/to/model",
        }
        insert_model_row(temp_db, 2, base_row)

        df = build_viability_report(temp_db, ["1d"], "6m")

        expected_cols = [
            "strategy_name", "assets", "asset_class", "interval", "backtest_period",
            "oracle_sharpe", "oracle_signals", "oracle_win_rate", "oracle_avg_pnl_per_trade", "oracle_run_id",
            "base_sharpe", "base_signals", "base_win_rate", "base_avg_pnl_per_trade", "base_run_id", "base_model_path",
            "signals_per_week", "viable",
        ]
        assert list(df.columns) == expected_cols, f"Got: {list(df.columns)}"

    def test_report_latest_run_wins(self, temp_db):
        """Test that only the latest run_id per key is used."""
        # Insert two oracle runs for the same strategy/assets/interval/backtest_period
        oracle_row_1 = {
            "strategy_name": "test_strat",
            "sharpe": 1.0,
            "signal_count": 5,
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, oracle_row_1)

        oracle_row_2 = {
            "strategy_name": "test_strat",
            "sharpe": 2.0,  # Higher sharpe in run_id 2
            "signal_count": 15,  # More signals in run_id 2
            "win_rate": 0.7,
            "avg_pnl_per_trade": 0.03,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 2, oracle_row_2)

        base_row = {
            "stage": "base",
            "strategy_name": "test_strat",
            "sharpe": 1.5,
            "signal_count": 12,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 3, base_row)

        df = build_viability_report(temp_db, ["1d"], "6m")

        assert len(df) == 1
        assert df.iloc[0]["oracle_sharpe"] == 2.0  # Latest run_id 2
        assert df.iloc[0]["oracle_signals"] == 15  # Latest run_id 2
        assert df.iloc[0]["oracle_run_id"] == 2

    def test_report_outer_join_nan_viable_false(self, temp_db):
        """Test outer join: strategy only in oracle, only in base."""
        # Strategy only in oracle
        oracle_only = {
            "strategy_name": "oracle_only_strat",
            "sharpe": 1.0,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, oracle_only)

        # Strategy only in base
        base_only = {
            "stage": "base",
            "strategy_name": "base_only_strat",
            "sharpe": 1.5,
            "signal_count": 12,
            "win_rate": 0.65,
            "avg_pnl_per_trade": 0.025,
            "assets": "ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, base_only)

        # Both present
        both_oracle = {
            "strategy_name": "both_strat",
            "sharpe": 1.2,
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "SOL-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, both_oracle)

        both_base = {
            "stage": "base",
            "strategy_name": "both_strat",
            "sharpe": 1.1,
            "signal_count": 7,
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "SOL-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, both_base)

        df = build_viability_report(temp_db, ["1d"], "6m")

        assert len(df) == 3

        # oracle_only_strat: base_* should be NaN, viable should be False
        oracle_only_row = df[df["strategy_name"] == "oracle_only_strat"].iloc[0]
        assert pd.isna(oracle_only_row["base_sharpe"])
        assert pd.isna(oracle_only_row["base_signals"])
        assert oracle_only_row["viable"] == False

        # base_only_strat: oracle_* should be NaN, viable should be False
        base_only_row = df[df["strategy_name"] == "base_only_strat"].iloc[0]
        assert pd.isna(base_only_row["oracle_sharpe"])
        assert pd.isna(base_only_row["oracle_signals"])
        assert base_only_row["viable"] == False

        # both_strat: has both, check viable based on sharpe
        both_row = df[df["strategy_name"] == "both_strat"].iloc[0]
        assert not pd.isna(both_row["oracle_sharpe"])
        assert not pd.isna(both_row["base_sharpe"])

    def test_report_viability_gating(self, temp_db):
        """Test viable flag with different sharpe and signal thresholds."""
        # Strategy with high sharpe and signals
        high_perf = {
            "strategy_name": "high_perf",
            "sharpe": 2.0,
            "signal_count": 20,
            "win_rate": 0.7,
            "avg_pnl_per_trade": 0.05,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, high_perf)

        high_perf_base = {
            "stage": "base",
            "strategy_name": "high_perf",
            "sharpe": 1.8,
            "signal_count": 18,
            "win_rate": 0.65,
            "avg_pnl_per_trade": 0.04,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, high_perf_base)

        # Strategy with low oracle sharpe (should fail viability)
        low_oracle = {
            "strategy_name": "low_oracle",
            "sharpe": -0.5,
            "signal_count": 15,
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, low_oracle)

        low_oracle_base = {
            "stage": "base",
            "strategy_name": "low_oracle",
            "sharpe": 1.0,
            "signal_count": 12,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, low_oracle_base)

        # Strategy with insufficient signals
        low_signals = {
            "strategy_name": "low_signals",
            "sharpe": 1.5,
            "signal_count": 2,  # Less than default min_signals=3
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "SOL-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, low_signals)

        low_signals_base = {
            "stage": "base",
            "strategy_name": "low_signals",
            "sharpe": 1.4,
            "signal_count": 1,  # Even fewer
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "SOL-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, low_signals_base)

        # Test with default min_sharpe=0.0, min_signals=3
        df = build_viability_report(temp_db, ["1d"], "6m", min_sharpe=0.0, min_signals=3)

        high_perf_row = df[df["strategy_name"] == "high_perf"].iloc[0]
        assert high_perf_row["viable"] == True

        low_oracle_row = df[df["strategy_name"] == "low_oracle"].iloc[0]
        assert low_oracle_row["viable"] == False  # oracle_sharpe < min_sharpe

        low_signals_row = df[df["strategy_name"] == "low_signals"].iloc[0]
        assert low_signals_row["viable"] == False  # signals < min_signals

    def test_report_signals_per_week(self, temp_db):
        """Test signals_per_week calculation."""
        row = {
            "strategy_name": "test_strat",
            "sharpe": 1.0,
            "signal_count": 100,
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "1m",
        }
        insert_oracle_row(temp_db, 1, row)

        base_row = {
            "stage": "base",
            "strategy_name": "test_strat",
            "sharpe": 1.0,
            "signal_count": 100,
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "1m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, base_row)

        df = build_viability_report(temp_db, ["1d"], "1m")

        # 1m = 365.25/12 / 7 ≈ 4.348 weeks
        expected_signals_per_week = 100.0 / _period_to_weeks("1m")
        actual_signals_per_week = df.iloc[0]["signals_per_week"]

        assert abs(actual_signals_per_week - expected_signals_per_week) < 1e-6

    def test_report_signals_per_week_fallback_to_oracle(self, temp_db):
        """Test that signals_per_week falls back to oracle_signals when base is NaN."""
        oracle_row = {
            "strategy_name": "fallback_strat",
            "sharpe": 1.0,
            "signal_count": 50,
            "win_rate": 0.5,
            "avg_pnl_per_trade": 0.01,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "2w",
        }
        insert_oracle_row(temp_db, 1, oracle_row)

        # No base row, so base signals will be NaN

        df = build_viability_report(temp_db, ["1d"], "2w")

        # signals_per_week should come from oracle (50 signals / 2 weeks = 25)
        expected_signals_per_week = 50.0 / _period_to_weeks("2w")
        actual_signals_per_week = df.iloc[0]["signals_per_week"]

        assert abs(actual_signals_per_week - expected_signals_per_week) < 1e-6

    def test_report_sort_order(self, temp_db):
        """Test sort order: viable first, then base_sharpe descending."""
        # Viable strategy with low base_sharpe
        viable_low = {
            "strategy_name": "viable_low",
            "sharpe": 2.0,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, viable_low)
        viable_low_base = {
            "stage": "base",
            "strategy_name": "viable_low",
            "sharpe": 1.0,  # Lower base sharpe
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, viable_low_base)

        # Viable strategy with high base_sharpe
        viable_high = {
            "strategy_name": "viable_high",
            "sharpe": 2.0,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, viable_high)
        viable_high_base = {
            "stage": "base",
            "strategy_name": "viable_high",
            "sharpe": 2.0,  # Higher base sharpe
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "ETH-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, viable_high_base)

        # Non-viable strategy
        not_viable = {
            "strategy_name": "not_viable",
            "sharpe": -1.0,
            "signal_count": 5,
            "win_rate": 0.4,
            "avg_pnl_per_trade": -0.01,
            "assets": "SOL-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, not_viable)
        not_viable_base = {
            "stage": "base",
            "strategy_name": "not_viable",
            "sharpe": 0.5,
            "signal_count": 4,
            "win_rate": 0.45,
            "avg_pnl_per_trade": 0.005,
            "assets": "SOL-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, not_viable_base)

        df = build_viability_report(temp_db, ["1d"], "6m")

        # viable_high should be first (viable=True, base_sharpe=2.0)
        # viable_low should be second (viable=True, base_sharpe=1.0)
        # not_viable should be last (viable=False)
        assert df.iloc[0]["strategy_name"] == "viable_high"
        assert df.iloc[1]["strategy_name"] == "viable_low"
        assert df.iloc[2]["strategy_name"] == "not_viable"

    def test_report_persistence(self, temp_db):
        """Test persistence to viability_report table and CSV."""
        oracle_row = {
            "strategy_name": "persist_strat",
            "sharpe": 1.5,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
        }
        insert_oracle_row(temp_db, 1, oracle_row)

        base_row = {
            "stage": "base",
            "strategy_name": "persist_strat",
            "sharpe": 1.2,
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": "/path/to/model",
        }
        insert_model_row(temp_db, 2, base_row)

        df = build_viability_report(temp_db, ["1d"], "6m")

        # Insert into viability_report table
        run_id = 100
        for _, row in df.iterrows():
            temp_db.execute(
                """INSERT INTO viability_report
                   (run_id, strategy_name, assets, asset_class, interval, backtest_period,
                    oracle_sharpe, oracle_signals, oracle_win_rate, oracle_avg_pnl_per_trade, oracle_run_id,
                    base_sharpe, base_signals, base_win_rate, base_avg_pnl_per_trade, base_run_id, base_model_path,
                    signals_per_week, viable)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, row["strategy_name"], row["assets"], row["asset_class"], row["interval"], row["backtest_period"],
                 row["oracle_sharpe"], row["oracle_signals"], row["oracle_win_rate"], row["oracle_avg_pnl_per_trade"], row["oracle_run_id"],
                 row["base_sharpe"], row["base_signals"], row["base_win_rate"], row["base_avg_pnl_per_trade"], row["base_run_id"], row["base_model_path"],
                 row["signals_per_week"], int(row["viable"])),
            )
        temp_db.commit()

        # Check row count in table
        table_rows = temp_db.execute("SELECT COUNT(*) FROM viability_report WHERE run_id = ?", (run_id,)).fetchone()[0]
        assert table_rows == len(df)
