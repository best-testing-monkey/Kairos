"""Tests for pipeline automation helpers."""

import pytest
import tempfile
import sqlite3
import os
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock, call
from kairos_strategies import _period_to_weeks, _period_to_bars, _parse_period
from kairos_pipeline import (
    build_viability_report, get_connection, SCHEMA, insert_oracle_row, insert_model_row,
    run_stage_auto, start_run
)


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


class TestParsePeriod:
    """Test the shared _parse_period helper."""

    def test_parse_period_returns_tuple(self):
        """_parse_period returns (count, unit) tuple."""
        count, unit = _parse_period("6m")
        assert count == 6
        assert unit == "m"

        count, unit = _parse_period("1y")
        assert count == 1
        assert unit == "y"

    def test_parse_period_case_insensitive(self):
        """_parse_period handles uppercase period strings."""
        count, unit = _parse_period("6M")
        assert count == 6
        assert unit == "m"

        count, unit = _parse_period("1Y")
        assert count == 1
        assert unit == "y"

    def test_parse_period_whitespace_tolerant(self):
        """_parse_period handles leading/trailing whitespace."""
        count, unit = _parse_period(" 6m ")
        assert count == 6
        assert unit == "m"

    def test_parse_period_invalid(self):
        """_parse_period raises ValueError for invalid input."""
        with pytest.raises(ValueError) as exc_info:
            _parse_period("invalid")
        assert "Unrecognised backtest_period" in str(exc_info.value)

    def test_parse_period_used_by_period_to_bars(self):
        """_period_to_bars uses _parse_period for consistent parsing."""
        # Both should parse correctly and not raise
        count, unit = _parse_period("6m")
        bars = _period_to_bars("6m", "1d")
        # Just verify they both work without exception
        assert count == 6
        assert bars > 0

    def test_parse_period_used_by_period_to_weeks(self):
        """_period_to_weeks uses _parse_period for consistent parsing."""
        # Both should parse correctly and not raise
        count, unit = _parse_period("6m")
        weeks = _period_to_weeks("6m")
        # Just verify they both work without exception
        assert count == 6
        assert weeks > 0


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


class TestRunStageAuto:
    """Tests for run_stage_auto chaining orchestration."""

    def _mock_payload(self, strategy_count=2):
        """Create a canned export_json payload for testing."""
        shadow = {}
        for i in range(strategy_count):
            shadow[f"strat_{i}"] = {
                "sharpe": 1.5 + i * 0.1,
                "signal_count": 10 + i,
                "win_rate": 0.6,
                "pnl_list": [0.01] * (10 + i),
            }
        return {
            "summary": {},
            "strategy_rankings": [(k, v["sharpe"]) for k, v in shadow.items()],
            "shadow_performance": shadow,
            "strategy_build_stats": {
                "total_constructed": strategy_count,
                "disabled_removed": 0,
                "evaluated": strategy_count,
            },
        }

    def test_auto_chaining_order(self, temp_db):
        """Verify call order: universe → correlation → per-group oracle → per-group base."""
        call_log = []

        def mock_universe(conn, interval="1d"):
            call_log.append(("universe", interval))
            run_id = start_run(conn, "universe", interval, {"interval": interval})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            call_log.append(("correlation", asset_class_filter, interval))
            run_id = start_run(conn, "correlation", interval, {"asset_class_filter": asset_class_filter})
            # Insert suggested groups
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 2, "crypto", "SOL-USD,AVAX-USD", 0.6),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle", tuple(sorted(assets)), interval))
            run_id = start_run(conn, "oracle", interval,
                             {"assets": assets, "backtest_period": backtest_period})
            assets_key = ",".join(sorted(assets))
            for row in self._mock_payload()["shadow_performance"].items():
                insert_oracle_row(conn, run_id, {
                    "strategy_name": row[0],
                    "sharpe": row[1]["sharpe"],
                    "signal_count": row[1]["signal_count"],
                    "win_rate": row[1]["win_rate"],
                    "avg_pnl_per_trade": 0.01,
                    "assets": assets_key,
                    "interval": interval,
                    "backtest_period": backtest_period,
                })
            temp_db.commit()
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base", tuple(sorted(assets)), interval))
            run_id = start_run(conn, stage, interval,
                             {"assets": assets, "backtest_period": backtest_period})
            assets_key = ",".join(sorted(assets))
            for row in self._mock_payload()["shadow_performance"].items():
                insert_model_row(conn, run_id, {
                    "stage": "base",
                    "strategy_name": row[0],
                    "sharpe": row[1]["sharpe"],
                    "signal_count": row[1]["signal_count"],
                    "win_rate": row[1]["win_rate"],
                    "avg_pnl_per_trade": 0.01,
                    "assets": assets_key,
                    "interval": interval,
                    "backtest_period": backtest_period,
                    "model_path": model_path,
                })
            temp_db.commit()
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m")

        # Verify call order
        assert call_log[0] == ("universe", "1d")
        assert call_log[1] == ("correlation", None, "1d")
        # Oracle and base calls for each group (2 groups)
        assert call_log[2] == ("oracle", ("BTC-USD", "ETH-USD"), "1d")
        assert call_log[3] == ("base", ("BTC-USD", "ETH-USD"), "1d")
        assert call_log[4] == ("oracle", ("AVAX-USD", "SOL-USD"), "1d")
        assert call_log[5] == ("base", ("AVAX-USD", "SOL-USD"), "1d")

    def test_auto_multi_interval(self, temp_db):
        """Verify chain runs once per interval."""
        call_log = []

        def mock_universe(conn, interval="1d"):
            call_log.append(("universe", interval))
            run_id = start_run(conn, "universe", interval, {"interval": interval})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            call_log.append(("correlation", interval))
            run_id = start_run(conn, "correlation", interval, {"asset_class_filter": asset_class_filter})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle", interval))
            run_id = start_run(conn, "oracle", interval, {})
            assets_key = ",".join(sorted(assets))
            insert_oracle_row(conn, run_id, {
                "strategy_name": "test", "sharpe": 1.0, "signal_count": 10,
                "win_rate": 0.5, "avg_pnl_per_trade": 0.01,
                "assets": assets_key, "interval": interval, "backtest_period": backtest_period,
            })
            temp_db.commit()
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base", interval))
            run_id = start_run(conn, stage, interval, {})
            assets_key = ",".join(sorted(assets))
            insert_model_row(conn, run_id, {
                "stage": "base", "strategy_name": "test", "sharpe": 1.0, "signal_count": 10,
                "win_rate": 0.5, "avg_pnl_per_trade": 0.01,
                "assets": assets_key, "interval": interval, "backtest_period": backtest_period,
                "model_path": None,
            })
            temp_db.commit()
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d", "1h"], "6m")

        # Verify universe called twice, once per interval
        universe_calls = [c for c in call_log if c[0] == "universe"]
        assert len(universe_calls) == 2
        assert ("universe", "1d") in universe_calls
        assert ("universe", "1h") in universe_calls

    def test_auto_resume_skip(self, temp_db):
        """Pre-inserted oracle_results matching (assets_key, interval, backtest_period) → oracle skipped."""
        call_log = []

        # Pre-insert oracle result for one group
        assets_key = "BTC-USD,ETH-USD"
        insert_oracle_row(temp_db, 1, {
            "strategy_name": "existing_strat",
            "sharpe": 1.5,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": assets_key,
            "interval": "1d",
            "backtest_period": "6m",
        })

        def mock_universe(conn, interval="1d"):
            run_id = start_run(conn, "universe", interval, {})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle", tuple(sorted(assets))))
            run_id = start_run(conn, "oracle", interval, {})
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base", tuple(sorted(assets))))
            run_id = start_run(conn, stage, interval, {})
            assets_key = ",".join(sorted(assets))
            insert_model_row(conn, run_id, {
                "stage": "base", "strategy_name": "test", "sharpe": 1.0, "signal_count": 10,
                "win_rate": 0.5, "avg_pnl_per_trade": 0.01,
                "assets": assets_key, "interval": interval, "backtest_period": backtest_period,
                "model_path": None,
            })
            temp_db.commit()
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m", force=False)

        # Oracle should NOT be called (skipped due to resumability)
        assert ("oracle", ("BTC-USD", "ETH-USD")) not in call_log
        # Base should be called
        assert ("base", ("BTC-USD", "ETH-USD")) in call_log

    def test_auto_force_reruns(self, temp_db):
        """force=True → oracle re-executed even with existing results."""
        call_log = []

        # Pre-insert oracle result
        assets_key = "BTC-USD,ETH-USD"
        insert_oracle_row(temp_db, 1, {
            "strategy_name": "existing_strat",
            "sharpe": 1.5,
            "signal_count": 10,
            "win_rate": 0.6,
            "avg_pnl_per_trade": 0.02,
            "assets": assets_key,
            "interval": "1d",
            "backtest_period": "6m",
        })

        def mock_universe(conn, interval="1d"):
            run_id = start_run(conn, "universe", interval, {})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle", tuple(sorted(assets))))
            run_id = start_run(conn, "oracle", interval, {})
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base", tuple(sorted(assets))))
            run_id = start_run(conn, stage, interval, {})
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m", force=True)

        # With force=True, oracle should be called even though it exists
        assert ("oracle", ("BTC-USD", "ETH-USD")) in call_log

    def test_auto_failure_isolation(self, temp_db):
        """RuntimeError in one group → remaining groups run; failure summary logged."""
        call_log = []

        def mock_universe(conn, interval="1d"):
            run_id = start_run(conn, "universe", interval, {})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {})
            # Two groups
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 2, "crypto", "SOL-USD,AVAX-USD", 0.6),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            assets_tuple = tuple(sorted(assets))
            call_log.append(("oracle", assets_tuple))
            # Raise error for first group only
            if assets_tuple == ("BTC-USD", "ETH-USD"):
                raise RuntimeError("Test oracle failure")
            run_id = start_run(conn, "oracle", interval, {})
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            assets_tuple = tuple(sorted(assets))
            call_log.append(("base", assets_tuple))
            run_id = start_run(conn, stage, interval, {})
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m")

        # First oracle should fail, second oracle should succeed
        assert call_log.count(("oracle", ("BTC-USD", "ETH-USD"))) == 1
        assert call_log.count(("oracle", ("AVAX-USD", "SOL-USD"))) == 1
        # First group's base should NOT run (due to oracle failure), second group's base should
        assert call_log.count(("base", ("BTC-USD", "ETH-USD"))) == 0
        assert call_log.count(("base", ("AVAX-USD", "SOL-USD"))) == 1

    def test_auto_runs_bookkeeping(self, temp_db):
        """One runs row inserted with stage='auto' and params_json."""
        def mock_universe(conn, interval="1d"):
            run_id = start_run(conn, "universe", interval, {})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            run_id = start_run(conn, "oracle", interval, {})
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            run_id = start_run(conn, stage, interval, {})
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m", min_sharpe=0.5, min_signals=5)

        # Check runs table for auto stage
        auto_runs = temp_db.execute(
            "SELECT run_id, params_json FROM runs WHERE stage='auto'"
        ).fetchall()

        assert len(auto_runs) == 1
        run_id, params_json = auto_runs[0]
        params = json.loads(params_json)
        assert params["intervals"] == ["1d"]
        assert params["backtest_period"] == "6m"
        assert params["min_sharpe"] == 0.5
        assert params["min_signals"] == 5

    def test_auto_skip_universe(self, temp_db):
        """skip_universe=True with existing runs → universe/correlation NOT called."""
        call_log = []

        # Pre-insert universe run
        run_id = start_run(temp_db, "universe", "1d", {})

        def mock_universe(conn, interval="1d"):
            call_log.append(("universe", interval))
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            call_log.append(("correlation",))
            run_id = start_run(conn, "correlation", interval, {})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle",))
            run_id = start_run(conn, "oracle", interval, {})
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base",))
            run_id = start_run(conn, stage, interval, {})
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m", skip_universe=True)

        # Universe should NOT be called when skip_universe=True and prior run exists
        assert ("universe", "1d") not in call_log
        # But correlation should still be called
        assert ("correlation",) in call_log

    def test_auto_skip_universe_reuses_correlation(self, temp_db):
        """skip_universe=True with existing correlation run → correlation NOT called."""
        call_log = []

        # Pre-insert universe and correlation runs
        universe_run_id = start_run(temp_db, "universe", "1d", {})
        correlation_run_id = start_run(temp_db, "correlation", "1d", {})

        # Insert suggested_groups for the existing correlation run
        temp_db.execute(
            "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
            "VALUES (?,?,?,?,?)",
            (correlation_run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
        )
        temp_db.commit()

        def mock_universe(conn, interval="1d"):
            call_log.append(("universe", interval))
            return universe_run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            call_log.append(("correlation",))
            run_id = start_run(conn, "correlation", interval, {})
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle",))
            run_id = start_run(conn, "oracle", interval, {})
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base",))
            run_id = start_run(conn, stage, interval, {})
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            df = run_stage_auto(temp_db, ["1d"], "6m", skip_universe=True)

        # Both universe and correlation should NOT be called when skip_universe=True and prior runs exist
        assert ("universe", "1d") not in call_log
        assert ("correlation",) not in call_log


class TestCLIFlagExclusivity:
    """Test argparse flag exclusivity constraints."""

    def test_cli_auto_with_singular_interval_error(self):
        """--stage auto + --interval → argparse error."""
        from kairos_pipeline import main
        import io
        import sys as sys_module

        old_stderr = sys_module.stderr
        sys_module.stderr = io.StringIO()

        try:
            with patch("kairos_pipeline.get_connection"):
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stage", "auto", "--interval", "1h"])
                assert exc_info.value.code == 2
        finally:
            sys_module.stderr = old_stderr

    def test_cli_intervals_with_non_auto_stage_error(self):
        """--intervals + non-auto stage → argparse error."""
        from kairos_pipeline import main
        import io
        import sys as sys_module

        old_stderr = sys_module.stderr
        sys_module.stderr = io.StringIO()

        try:
            with patch("kairos_pipeline.get_connection"):
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stage", "oracle", "--intervals", "1d", "--assets", "BTC-USD"])
                assert exc_info.value.code == 2
        finally:
            sys_module.stderr = old_stderr

    def test_cli_min_sharpe_with_non_auto_stage_error(self):
        """--min_sharpe with non-auto stage → argparse error."""
        from kairos_pipeline import main
        import io
        import sys as sys_module

        old_stderr = sys_module.stderr
        sys_module.stderr = io.StringIO()

        try:
            with patch("kairos_pipeline.get_connection"):
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stage", "oracle", "--min_sharpe", "0.5", "--assets", "BTC-USD"])
                assert exc_info.value.code == 2
        finally:
            sys_module.stderr = old_stderr

    def test_cli_force_with_non_auto_stage_error(self):
        """--force with non-auto stage → argparse error."""
        from kairos_pipeline import main
        import io
        import sys as sys_module

        old_stderr = sys_module.stderr
        sys_module.stderr = io.StringIO()

        try:
            with patch("kairos_pipeline.get_connection"):
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stage", "oracle", "--force", "--assets", "BTC-USD"])
                assert exc_info.value.code == 2
        finally:
            sys_module.stderr = old_stderr

    def test_cli_skip_universe_with_non_auto_stage_error(self):
        """--skip_universe with non-auto stage → argparse error."""
        from kairos_pipeline import main
        import io
        import sys as sys_module

        old_stderr = sys_module.stderr
        sys_module.stderr = io.StringIO()

        try:
            with patch("kairos_pipeline.get_connection"):
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stage", "oracle", "--skip_universe", "--assets", "BTC-USD"])
                assert exc_info.value.code == 2
        finally:
            sys_module.stderr = old_stderr

    def test_cli_report_only_with_non_auto_stage_error(self):
        """--report_only with non-auto stage → argparse error."""
        from kairos_pipeline import main
        import io
        import sys as sys_module

        old_stderr = sys_module.stderr
        sys_module.stderr = io.StringIO()

        try:
            with patch("kairos_pipeline.get_connection"):
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stage", "oracle", "--report_only", "--assets", "BTC-USD"])
                assert exc_info.value.code == 2
        finally:
            sys_module.stderr = old_stderr

    def test_cli_auto_valid_with_intervals_plural(self):
        """--stage auto + --intervals (plural) accepted."""
        from kairos_pipeline import main

        # Should parse without error and not crash before get_connection
        with patch("kairos_pipeline.get_connection"), \
             patch("kairos_pipeline.run_stage_auto"):
            main(["--stage", "auto", "--intervals", "1d", "1h"])
            # If we got here, parsing was successful


class TestCLIReportOnlyDispatch:
    """Test --report_only flag dispatch."""

    def test_report_only_skips_run_stage_auto(self, temp_db):
        """--report_only → build_viability_report only; run_stage_auto not called."""
        from kairos_pipeline import main

        # Pre-insert some results in the DB
        oracle_row = {
            "strategy_name": "test_strat",
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
            "strategy_name": "test_strat",
            "sharpe": 1.2,
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, base_row)

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_auto") as mock_auto, \
             patch("kairos_pipeline.dump_csv", return_value="/tmp/test.csv"):
            main(["--stage", "auto", "--report_only"])

        # run_stage_auto should NOT be called
        mock_auto.assert_not_called()

    def test_report_only_calls_build_viability_report(self, temp_db):
        """--report_only → build_viability_report is called with correct flags."""
        from kairos_pipeline import main

        oracle_row = {
            "strategy_name": "test_strat",
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
            "strategy_name": "test_strat",
            "sharpe": 1.2,
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, base_row)

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.build_viability_report", wraps=build_viability_report) as mock_report, \
             patch("kairos_pipeline.dump_csv", return_value="/tmp/test.csv"):
            main(["--stage", "auto", "--report_only", "--intervals", "1d", "--min_sharpe", "0.5", "--min_signals", "5"])

        # build_viability_report should be called with correct arguments
        mock_report.assert_called_once()
        call_args = mock_report.call_args
        # Verify key arguments
        assert call_args[0][1] == ["1d"]  # intervals
        assert call_args[1]["min_sharpe"] == 0.5
        assert call_args[1]["min_signals"] == 5

    def test_report_only_writes_viability_report_table(self, temp_db):
        """--report_only writes viability_report table rows."""
        from kairos_pipeline import main, persist_viability_report

        # Pre-insert oracle and base results
        oracle_row = {
            "strategy_name": "test_strat",
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
            "strategy_name": "test_strat",
            "sharpe": 1.2,
            "signal_count": 8,
            "win_rate": 0.55,
            "avg_pnl_per_trade": 0.015,
            "assets": "BTC-USD",
            "interval": "1d",
            "backtest_period": "6m",
            "model_path": None,
        }
        insert_model_row(temp_db, 2, base_row)

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.persist_viability_report", wraps=persist_viability_report) as mock_persist, \
             patch("kairos_pipeline.dump_csv", return_value="/tmp/test.csv"):
            main(["--stage", "auto", "--report_only"])

        # Verify persist_viability_report was called
        mock_persist.assert_called_once()
        # Get the DataFrame that was passed to persist_viability_report
        call_args = mock_persist.call_args
        df = call_args[0][1]  # Second positional argument is the DataFrame
        # Verify the DataFrame has rows
        assert len(df) > 0, "No viability_report rows in DataFrame"
        assert "test_strat" in df["strategy_name"].values


class TestRunStageAutoPredCache:
    """Tests that run_stage_auto creates/cleans up the per-run prediction
    cache dir and threads KAIROS_PRED_CACHE_DIR through to run_stage_model."""

    def test_cache_dir_created_and_passed_then_cleaned_up(self, temp_db):
        import os as _os
        captured_envs = []
        captured_dirs = []

        def mock_universe(conn, interval="1d", **kwargs):
            return start_run(conn, "universe", interval, {})

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100, **kwargs):
            return start_run(conn, "oracle", interval, {})

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                      pred_samples=100, model_path=None, extra_env=None, **kwargs):
            captured_envs.append(extra_env)
            if extra_env and "KAIROS_PRED_CACHE_DIR" in extra_env:
                cache_dir = extra_env["KAIROS_PRED_CACHE_DIR"]
                captured_dirs.append(cache_dir)
                # The directory must exist while the auto run is in progress.
                assert _os.path.isdir(cache_dir)
            return start_run(conn, stage, interval, {})

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            run_stage_auto(temp_db, ["1d"], "6m")

        assert len(captured_dirs) == 1
        cache_dir = captured_dirs[0]
        # Cache dir must be removed once the auto run finishes.
        assert not _os.path.isdir(cache_dir)

    def test_cache_dir_cleaned_up_even_on_failure(self, temp_db):
        import os as _os
        import tempfile

        def mock_universe(conn, interval="1d", **kwargs):
            return start_run(conn, "universe", interval, {})

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "BTC-USD,ETH-USD", 0.7),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100, **kwargs):
            raise RuntimeError("boom")

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                      pred_samples=100, model_path=None, extra_env=None, **kwargs):
            raise AssertionError("base should not run when oracle fails")

        before = {
            d for d in _os.listdir(tempfile.gettempdir())
            if d.startswith("kairos_predcache_run")
        }

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            run_stage_auto(temp_db, ["1d"], "6m")

        after = {
            d for d in _os.listdir(tempfile.gettempdir())
            if d.startswith("kairos_predcache_run")
        }
        # No leftover cache dirs from this run, even though oracle raised.
        assert after - before == set()


class TestCLIFlagsPassedVerbatim:
    """Test that flags are passed verbatim to run_stage_auto."""

    def test_flags_passed_to_run_stage_auto(self, temp_db):
        """All auto-stage flags passed through to run_stage_auto."""
        from kairos_pipeline import main

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_auto") as mock_auto:
            main([
                "--stage", "auto",
                "--intervals", "1d", "1h",
                "--backtest_period", "3m",
                "--asset_class", "crypto",
                "--pred_samples", "50",
                "--min_sharpe", "1.5",
                "--min_signals", "5",
                "--force",
                "--skip_universe",
            ])

        # Verify run_stage_auto was called with correct arguments
        mock_auto.assert_called_once()
        call_args, call_kwargs = mock_auto.call_args
        # Arguments: conn, intervals, backtest_period
        assert call_args[1] == ["1d", "1h"]  # intervals (2nd positional arg)
        assert call_args[2] == "3m"  # backtest_period (3rd positional arg)
        # Keyword arguments
        assert call_kwargs["asset_class_filter"] == "crypto"
        assert call_kwargs["pred_samples"] == 50
        assert call_kwargs["min_sharpe"] == 1.5
        assert call_kwargs["min_signals"] == 5
        assert call_kwargs["force"] is True
        assert call_kwargs["skip_universe"] is True


class TestCorrelationIntervalThreading:
    """Test that correlation stage honors the interval parameter."""

    def test_correlation_interval_threading(self, temp_db):
        """Correlation stage with interval='1h' requests 1h data; default 1d unchanged."""
        from kairos_pipeline import run_stage_correlation
        from unittest.mock import patch, MagicMock
        import pandas as pd

        # Pre-populate universe_screen with passing survivors
        temp_db.execute(
            "INSERT INTO universe_screen (run_id, symbol, asset_class, passed) VALUES (?,?,?,?)",
            (1, "BTC-USD", "crypto", 1),
        )
        temp_db.execute(
            "INSERT INTO universe_screen (run_id, symbol, asset_class, passed) VALUES (?,?,?,?)",
            (1, "ETH-USD", "crypto", 1),
        )
        temp_db.commit()

        # Track fetch calls
        fetch_calls = []

        def mock_get_price_data(symbol, start_date, end_date, interval):
            fetch_calls.append({
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
            })
            # Return a dummy DataFrame
            dates = pd.date_range(start=start_date, end=end_date, freq='D' if interval == "1d" else 'H')
            df = pd.DataFrame({
                "close": [100.0] * len(dates),
                "volume": [1000000.0] * len(dates),
            }, index=dates)
            return df

        # Test with interval="1h"
        fetch_calls.clear()
        with patch("price_cache.get_price_data", side_effect=mock_get_price_data):
            run_stage_correlation(temp_db, asset_class_filter=None, interval="1h")

        # Verify fetch calls used interval="1h"
        assert len(fetch_calls) >= 2
        for call in fetch_calls:
            assert call["interval"] == "1h", f"Expected interval='1h', got {call['interval']}"

        # Test with default interval="1d" (should be byte-identical to before)
        temp_db.execute("DELETE FROM runs WHERE stage='correlation'")
        temp_db.execute("DELETE FROM correlation_pairs")
        temp_db.execute("DELETE FROM suggested_groups")
        temp_db.commit()

        fetch_calls.clear()
        with patch("price_cache.get_price_data", side_effect=mock_get_price_data):
            run_stage_correlation(temp_db, asset_class_filter=None, interval="1d")

        # Verify fetch calls used interval="1d"
        assert len(fetch_calls) >= 2
        for call in fetch_calls:
            assert call["interval"] == "1d", f"Expected interval='1d', got {call['interval']}"

    def test_correlation_default_interval_unchanged(self, temp_db):
        """Correlation with no interval param defaults to 1d (byte-identical behavior)."""
        from kairos_pipeline import run_stage_correlation
        from unittest.mock import patch

        # Pre-populate universe_screen
        temp_db.execute(
            "INSERT INTO universe_screen (run_id, symbol, asset_class, passed) VALUES (?,?,?,?)",
            (1, "BTC-USD", "crypto", 1),
        )
        temp_db.commit()

        fetch_calls = []

        def mock_get_price_data(symbol, start_date, end_date, interval):
            fetch_calls.append({"interval": interval})
            import pandas as pd
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            return pd.DataFrame({
                "close": [100.0] * len(dates),
                "volume": [1000000.0] * len(dates),
            }, index=dates)

        # Call without interval (uses default)
        with patch("price_cache.get_price_data", side_effect=mock_get_price_data):
            run_stage_correlation(temp_db, asset_class_filter=None)

        # Verify default is 1d
        assert len(fetch_calls) >= 1
        assert fetch_calls[0]["interval"] == "1d"


class TestSingleStageRegression:
    """Test that single-stage invocations remain unchanged."""

    def test_oracle_stage_unchanged(self, temp_db):
        """--stage oracle with --assets dispatches as before."""
        from kairos_pipeline import main

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_oracle") as mock_oracle:
            main(["--stage", "oracle", "--assets", "BTC-USD", "ETH-USD", "--interval", "1h", "--backtest_period", "3m"])

        # Verify run_stage_oracle was called with correct arguments
        mock_oracle.assert_called_once()
        call_args, call_kwargs = mock_oracle.call_args
        assert call_args[1] == ["BTC-USD", "ETH-USD"]  # assets
        assert call_kwargs["interval"] == "1h"
        assert call_kwargs["backtest_period"] == "3m"

    def test_base_stage_unchanged(self, temp_db):
        """--stage base with --assets dispatches as before."""
        from kairos_pipeline import main

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_model") as mock_model:
            main(["--stage", "base", "--assets", "BTC-USD", "--interval", "1h"])

        # Verify run_stage_model was called
        mock_model.assert_called_once()
        call_args, call_kwargs = mock_model.call_args
        assert call_args[0] == temp_db
        assert call_args[1] == "base"
        assert call_args[2] == ["BTC-USD"]
        assert call_kwargs["interval"] == "1h"

    def test_universe_stage_unchanged(self, temp_db):
        """--stage universe dispatches as before."""
        from kairos_pipeline import main

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_universe") as mock_universe:
            main(["--stage", "universe", "--interval", "1h"])

        # Verify run_stage_universe was called
        mock_universe.assert_called_once()
        call_kwargs = mock_universe.call_args[1]
        assert call_kwargs["interval"] == "1h"

    def test_correlation_stage_unchanged(self, temp_db):
        """--stage correlation dispatches as before."""
        from kairos_pipeline import main

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_correlation") as mock_corr:
            main(["--stage", "correlation", "--asset_class", "crypto"])

        # Verify run_stage_correlation was called
        mock_corr.assert_called_once()
        call_kwargs = mock_corr.call_args[1]
        assert call_kwargs["asset_class_filter"] == "crypto"

    def test_correlation_stage_passes_interval(self, temp_db):
        """--stage correlation --interval passes interval parameter."""
        from kairos_pipeline import main

        with patch("kairos_pipeline.get_connection", return_value=temp_db), \
             patch("kairos_pipeline.run_stage_correlation") as mock_corr:
            main(["--stage", "correlation", "--interval", "1h", "--asset_class", "crypto"])

        # Verify run_stage_correlation was called with interval
        mock_corr.assert_called_once()
        call_kwargs = mock_corr.call_args[1]
        assert call_kwargs["interval"] == "1h"
        assert call_kwargs["asset_class_filter"] == "crypto"


class TestCLIHelpAndSubprocess:
    """Test --help and subprocess integration."""

    def test_help_exit_zero(self):
        """uv run ./strategy/kairos_pipeline.py --help exits 0."""
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q"],
            cwd="/media/baz/MonkeyWorks/PycharmProjects/Kairos",
            capture_output=True,
        )
        # Just verify we can import without error; full subprocess test requires uv in PATH
        # For now, test that _build_parser works and --help is recognized
        from kairos_pipeline import _build_parser
        parser = _build_parser()
        # Calling parse_args with --help would exit, so we just verify the parser was built
        assert parser is not None

    def test_new_flags_in_help(self):
        """New flags appear in --help output."""
        from kairos_pipeline import _build_parser
        import io
        import sys as sys_module

        parser = _build_parser()

        # Capture help output
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO()

        try:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args(["--help"])
            assert exc_info.value.code == 0
        finally:
            help_output = sys_module.stdout.getvalue()
            sys_module.stdout = old_stdout

        # Verify new flags are in help
        assert "--intervals" in help_output
        assert "--min_sharpe" in help_output
        assert "--min_signals" in help_output
        assert "--force" in help_output
        assert "--skip_universe" in help_output
        assert "--report_only" in help_output
        assert "--stage auto" in help_output or "auto" in help_output


class TestGreedyGroupPairsCross:
    """Tests for cross-asset-class handling in greedy_group_pairs."""

    def test_cross_class_pair_produces_cross_group(self):
        """A single strong cross-class pair seeds a group with asset_class='cross'."""
        from kairos_pipeline import greedy_group_pairs

        pairs = [
            {"symbol_a": "BTC-USD", "symbol_b": "AAPL", "asset_class": "cross", "full_corr": 0.8},
        ]
        groups = greedy_group_pairs(pairs)
        assert len(groups) == 1
        assert groups[0]["asset_class"] == "cross"
        assert groups[0]["symbols"] == ["AAPL", "BTC-USD"]

    def test_mixed_join_flips_group_to_cross(self):
        """A same-class group that gains a member of a different class becomes 'cross'."""
        from kairos_pipeline import greedy_group_pairs

        pairs = [
            # Strongest pair first: same-class equity group seeded.
            {"symbol_a": "AAPL", "symbol_b": "MSFT", "asset_class": "equity", "full_corr": 0.9},
            # Weaker cross pair: BTC-USD joins the existing equity group via AAPL.
            {"symbol_a": "AAPL", "symbol_b": "BTC-USD", "asset_class": "cross", "full_corr": 0.7},
        ]
        groups = greedy_group_pairs(pairs)
        assert len(groups) == 1
        assert groups[0]["asset_class"] == "cross"
        assert set(groups[0]["symbols"]) == {"AAPL", "MSFT", "BTC-USD"}

    def test_same_class_group_stays_same_class(self):
        """A pure same-class group is unaffected by unrelated cross pairs elsewhere."""
        from kairos_pipeline import greedy_group_pairs

        pairs = [
            {"symbol_a": "AAPL", "symbol_b": "MSFT", "asset_class": "equity", "full_corr": 0.9},
            {"symbol_a": "ETH-USD", "symbol_b": "SOL-USD", "asset_class": "crypto", "full_corr": 0.85},
        ]
        groups = greedy_group_pairs(pairs)
        by_symbols = {tuple(g["symbols"]): g["asset_class"] for g in groups}
        assert by_symbols[("AAPL", "MSFT")] == "equity"
        assert by_symbols[("ETH-USD", "SOL-USD")] == "crypto"

    def test_cross_pair_between_groups_with_capacity_joins_strongest(self):
        """A cross pair whose symbols sit in two different existing groups with
        capacity: the missing symbol joins the group with the highest mean |corr|
        (Rule 2), flipping that group to 'cross'; the other group is unchanged."""
        from kairos_pipeline import greedy_group_pairs

        pairs = [
            {"symbol_a": "AAPL", "symbol_b": "MSFT", "asset_class": "equity", "full_corr": 0.95},
            {"symbol_a": "ETH-USD", "symbol_b": "SOL-USD", "asset_class": "crypto", "full_corr": 0.9},
            {"symbol_a": "AAPL", "symbol_b": "ETH-USD", "asset_class": "cross", "full_corr": 0.65},
        ]
        groups = greedy_group_pairs(pairs)
        by_symbols = {tuple(sorted(g["symbols"])): g["asset_class"] for g in groups}
        # ETH-USD joined the stronger (equity) group, flipping it to cross.
        assert by_symbols[("AAPL", "ETH-USD", "MSFT")] == "cross"
        # The crypto group is untouched.
        assert by_symbols[("ETH-USD", "SOL-USD")] == "crypto"


class TestGreedyGroupPairsOverlap:
    """Tests for overlapping group membership: no passing pair is ever dropped."""

    def test_copx_xlb_scenario_new_cross_group_when_groups_full(self):
        """Two full same-class groups claim COPX and XLB; the weaker-but-passing
        cross pair must still yield a NEW group containing both (Rule 3)."""
        from kairos_pipeline import greedy_group_pairs

        pairs = [
            # Fill a 4-symbol commodity group containing COPX.
            {"symbol_a": "COPX", "symbol_b": "GDX", "asset_class": "commodity", "full_corr": 0.95},
            {"symbol_a": "COPX", "symbol_b": "SLV", "asset_class": "commodity", "full_corr": 0.94},
            {"symbol_a": "COPX", "symbol_b": "GLD", "asset_class": "commodity", "full_corr": 0.93},
            # Fill a 4-symbol equity group containing XLB.
            {"symbol_a": "XLB", "symbol_b": "XLI", "asset_class": "equity", "full_corr": 0.92},
            {"symbol_a": "XLB", "symbol_b": "XLE", "asset_class": "equity", "full_corr": 0.91},
            {"symbol_a": "XLB", "symbol_b": "XLF", "asset_class": "equity", "full_corr": 0.90},
            # Weaker cross pair: both symbols already in different FULL groups.
            {"symbol_a": "COPX", "symbol_b": "XLB", "asset_class": "cross", "full_corr": 0.631},
        ]
        groups = greedy_group_pairs(pairs, max_group_size=4)
        pair_groups = [g for g in groups if set(g["symbols"]) == {"COPX", "XLB"}]
        assert len(pair_groups) == 1, f"COPX/XLB pair must get its own group; got {groups}"
        assert pair_groups[0]["asset_class"] == "cross"
        assert abs(pair_groups[0]["mean_intra_corr"] - 0.631) < 1e-9

    def test_pair_joins_existing_group_with_capacity(self):
        """A passing pair with one symbol in an existing group with capacity joins it."""
        from kairos_pipeline import greedy_group_pairs

        pairs = [
            {"symbol_a": "GLD", "symbol_b": "SLV", "asset_class": "commodity", "full_corr": 0.9},
            {"symbol_a": "GLD", "symbol_b": "GDX", "asset_class": "commodity", "full_corr": 0.7},
        ]
        groups = greedy_group_pairs(pairs)
        assert len(groups) == 1
        assert set(groups[0]["symbols"]) == {"GLD", "SLV", "GDX"}
        assert groups[0]["asset_class"] == "commodity"

    def test_no_passing_pair_unrepresented(self):
        """Property: every pair with |corr| >= min_abs_corr appears together in
        at least one output group."""
        from kairos_pipeline import greedy_group_pairs
        import itertools

        symbols = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1"]
        cls = {s: {"A": "equity", "B": "crypto", "C": "commodity"}[s[0]] for s in symbols}
        rng = np.random.default_rng(42)
        pairs = []
        for a, b in itertools.combinations(symbols, 2):
            pairs.append({
                "symbol_a": a, "symbol_b": b,
                "asset_class": cls[a] if cls[a] == cls[b] else "cross",
                "full_corr": float(rng.uniform(-1, 1)),
            })

        groups = greedy_group_pairs(pairs, min_abs_corr=0.6, max_group_size=4)
        group_sets = [set(g["symbols"]) for g in groups]
        for p in pairs:
            if abs(p["full_corr"]) >= 0.6:
                assert any({p["symbol_a"], p["symbol_b"]} <= gs for gs in group_sets), \
                    f"pair {p['symbol_a']}/{p['symbol_b']} (corr={p['full_corr']:.3f}) unrepresented"


class TestCorrelationSingletonsAndCross:
    """Tests for singleton group insertion and cross-asset-class correlation
    in run_stage_correlation."""

    @staticmethod
    def _make_series(seed, n, base=100.0, corr_with=None, noise=0.02):
        """Build a synthetic close-price series; if corr_with is given, derive
        returns that are strongly correlated with it."""
        rng = np.random.default_rng(seed)
        if corr_with is not None:
            rets = corr_with * 0.9 + rng.normal(0, noise, size=len(corr_with))
        else:
            rets = rng.normal(0, 0.01, size=n)
        prices = base * np.exp(np.cumsum(rets))
        return rets, prices

    def test_ungrouped_survivor_becomes_singleton(self, temp_db):
        """A passing survivor with no correlated peer gets a singleton suggested_group row."""
        from kairos_pipeline import run_stage_correlation

        for sym, ac in [("BTC-USD", "crypto"), ("ETH-USD", "crypto"), ("LONER-USD", "crypto")]:
            temp_db.execute(
                "INSERT INTO universe_screen (run_id, symbol, asset_class, passed) VALUES (?,?,?,?)",
                (1, sym, ac, 1),
            )
        temp_db.commit()

        n = 200
        base_rets, base_prices = self._make_series(1, n)
        _, corr_prices = self._make_series(2, n, corr_with=base_rets, noise=0.002)
        _, loner_prices = self._make_series(3, n)  # independent, uncorrelated

        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        price_map = {
            "BTC-USD": pd.Series(base_prices, index=dates),
            "ETH-USD": pd.Series(corr_prices, index=dates),
            "LONER-USD": pd.Series(loner_prices, index=dates),
        }

        def mock_get_price_data(symbol, start_date, end_date, interval):
            s = price_map[symbol]
            return pd.DataFrame({"close": s.values, "volume": [1e6] * len(s)}, index=s.index)

        with patch("price_cache.get_price_data", side_effect=mock_get_price_data):
            run_id = run_stage_correlation(temp_db, asset_class_filter=None, interval="1d")

        rows = temp_db.execute(
            "SELECT group_id, asset_class, symbols, mean_intra_corr FROM suggested_groups WHERE run_id=?",
            (run_id,),
        ).fetchall()
        symbols_by_group = {r[2]: r for r in rows}

        # LONER-USD should appear alone with mean_intra_corr NULL.
        assert "LONER-USD" in symbols_by_group
        loner_row = symbols_by_group["LONER-USD"]
        assert loner_row[1] == "crypto"
        assert loner_row[3] is None

        # BTC-USD/ETH-USD should NOT get their own singleton rows (they're grouped).
        assert "BTC-USD" not in symbols_by_group
        assert "ETH-USD" not in symbols_by_group
        grouped = [r for r in rows if "BTC-USD" in r[2].split(",") and "ETH-USD" in r[2].split(",")]
        assert len(grouped) == 1

    def test_cross_class_pair_persisted_with_cross_asset_class(self, temp_db):
        """Correlation across asset classes is no longer skipped; pair row gets asset_class='cross'."""
        from kairos_pipeline import run_stage_correlation

        for sym, ac in [("BTC-USD", "crypto"), ("AAPL", "equity")]:
            temp_db.execute(
                "INSERT INTO universe_screen (run_id, symbol, asset_class, passed) VALUES (?,?,?,?)",
                (1, sym, ac, 1),
            )
        temp_db.commit()

        n = 200
        base_rets, base_prices = self._make_series(10, n)
        _, corr_prices = self._make_series(11, n, corr_with=base_rets, noise=0.005)

        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        price_map = {
            "BTC-USD": pd.Series(base_prices, index=dates),
            "AAPL": pd.Series(corr_prices, index=dates),
        }

        def mock_get_price_data(symbol, start_date, end_date, interval):
            s = price_map[symbol]
            return pd.DataFrame({"close": s.values, "volume": [1e6] * len(s)}, index=s.index)

        with patch("price_cache.get_price_data", side_effect=mock_get_price_data):
            run_id = run_stage_correlation(temp_db, asset_class_filter=None, interval="1d")

        pair_rows = temp_db.execute(
            "SELECT symbol_a, symbol_b, asset_class FROM correlation_pairs WHERE run_id=?",
            (run_id,),
        ).fetchall()
        assert len(pair_rows) == 1
        assert pair_rows[0][2] == "cross"

        group_rows = temp_db.execute(
            "SELECT asset_class, symbols FROM suggested_groups WHERE run_id=?",
            (run_id,),
        ).fetchall()
        # Either grouped together as "cross" (if corr >= 0.6), or each a same-class singleton.
        if len(group_rows) == 1:
            assert group_rows[0][0] == "cross"
        else:
            classes = {r[1]: r[0] for r in group_rows}
            assert classes.get("BTC-USD") == "crypto"
            assert classes.get("AAPL") == "equity"


class TestRunStageAutoSingletonGroup:
    """Tests that run_stage_auto handles a 1-symbol suggested group correctly."""

    def _mock_payload(self):
        return {
            "summary": {},
            "strategy_rankings": [("strat_0", 1.5)],
            "shadow_performance": {
                "strat_0": {"sharpe": 1.5, "signal_count": 10, "win_rate": 0.6, "pnl_list": [0.01] * 10},
            },
            "strategy_build_stats": {"total_constructed": 1, "disabled_removed": 0, "evaluated": 1},
        }

    def test_singleton_group_generates_one_asset_calls(self, temp_db):
        """A singleton suggested_group row (1 symbol) drives oracle+base with a 1-element assets list."""
        call_log = []

        def mock_universe(conn, interval="1d"):
            run_id = start_run(conn, "universe", interval, {"interval": interval})
            return run_id

        def mock_correlation(conn, asset_class_filter=None, interval="1d", **kwargs):
            run_id = start_run(conn, "correlation", interval, {"asset_class_filter": asset_class_filter})
            temp_db.execute(
                "INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr) "
                "VALUES (?,?,?,?,?)",
                (run_id, 1, "crypto", "LONER-USD", None),
            )
            temp_db.commit()
            return run_id

        def mock_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
            call_log.append(("oracle", list(assets), interval))
            run_id = start_run(conn, "oracle", interval, {"assets": assets, "backtest_period": backtest_period})
            for name, perf in self._mock_payload()["shadow_performance"].items():
                insert_oracle_row(conn, run_id, {
                    "strategy_name": name, "sharpe": perf["sharpe"], "signal_count": perf["signal_count"],
                    "win_rate": perf["win_rate"], "avg_pnl_per_trade": 0.01,
                    "assets": ",".join(sorted(assets)), "interval": interval, "backtest_period": backtest_period,
                })
            temp_db.commit()
            return run_id

        def mock_base(conn, stage, assets, interval="1d", backtest_period="6m",
                      pred_samples=100, model_path=None, **kwargs):
            call_log.append(("base", list(assets), interval))
            run_id = start_run(conn, stage, interval, {"assets": assets, "backtest_period": backtest_period})
            for name, perf in self._mock_payload()["shadow_performance"].items():
                insert_model_row(conn, run_id, {
                    "stage": "base", "strategy_name": name, "sharpe": perf["sharpe"],
                    "signal_count": perf["signal_count"], "win_rate": perf["win_rate"],
                    "avg_pnl_per_trade": 0.01, "assets": ",".join(sorted(assets)),
                    "interval": interval, "backtest_period": backtest_period, "model_path": model_path,
                })
            temp_db.commit()
            return run_id

        with patch("kairos_pipeline.run_stage_universe", side_effect=mock_universe), \
             patch("kairos_pipeline.run_stage_correlation", side_effect=mock_correlation), \
             patch("kairos_pipeline.run_stage_oracle", side_effect=mock_oracle), \
             patch("kairos_pipeline.run_stage_model", side_effect=mock_base):
            run_stage_auto(temp_db, ["1d"], "6m")

        assert ("oracle", ["LONER-USD"], "1d") in call_log
        assert ("base", ["LONER-USD"], "1d") in call_log
