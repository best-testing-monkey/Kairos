"""test_allocation.py — Unit tests for allocation.Candidate and fetch_signals().

Tests the schema and fetch adapter without GPU/network, using hand-constructed
fixtures that mirror the exact dict shapes from kairos_signals.py run().
"""

import csv
import math
import pytest
import tempfile
import os
from allocation import (
    Candidate, fetch_signals, validate_candidate, AllocationConfig, compute_derived,
    compute_ev_ratio, select_candidates, size_selected, allocate, load_cluster_map,
    AllocationResult
)


class TestCandidateDataclass:
    """Test Candidate schema and dataclass structure."""

    def test_candidate_all_fields(self):
        """Candidate has all required fields plus nullable avg_* fields."""
        c = Candidate(
            strategy="path_execution",
            ticker="REMX",
            direction="short",
            entry=79.73,
            stop=84.51,
            target=73.71,
            ev_pct=4.04,
            base_win_rate=0.47,
            n=161,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.23,
            advised_liquidity_pct=11.0,
            avg_win_pct=None,
            avg_loss_pct=None,
            avg_holding_days=None,
        )
        assert c.strategy == "path_execution"
        assert c.ticker == "REMX"
        assert c.direction == "short"
        assert c.entry == 79.73
        assert c.stop == 84.51
        assert c.target == 73.71
        assert c.ev_pct == 4.04
        assert c.base_win_rate == 0.47
        assert c.n == 161
        assert c.backtest_period == "2023-01-01 to 2023-12-31"
        assert c.sharpe == 1.23
        assert c.advised_liquidity_pct == 11.0
        assert c.avg_win_pct is None
        assert c.avg_loss_pct is None
        assert c.avg_holding_days is None

    def test_candidate_avg_fields_with_values(self):
        """Candidate avg_* fields can hold float values when provided."""
        c = Candidate(
            strategy="strategy1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
            avg_win_pct=2.5,
            avg_loss_pct=1.5,
            avg_holding_days=3.5,
        )
        assert c.avg_win_pct == 2.5
        assert c.avg_loss_pct == 1.5
        assert c.avg_holding_days == 3.5


class TestFetchSignals:
    """Test fetch_signals() adapter function."""

    def test_basic_long_signal(self):
        """Single LONG signal is converted to Candidate with direction='long'."""
        stats_rows = [
            {
                "strategy": "path_execution",
                "symbol": "BTC",
                "direction": "LONG",
                "entry": 50000.0,
                "stop": 48000.0,
                "target": 55000.0,
                "expected_value": 2000.0,
                "base_sharpe": 1.5,
                "base_win_rate": 0.55,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.10,
            }
        ]
        advice_rows = [
            {
                "expected_value": 2000.0,
                "entry": 50000.0,
                "base_win_rate": 0.55,
                "base_signals": 100,
                "oracle_signals": 120,
                "signal": "Buy BTC",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.strategy == "path_execution"
        assert c.ticker == "BTC"
        assert c.direction == "long"
        assert c.entry == 50000.0
        assert c.stop == 48000.0
        assert c.target == 55000.0
        assert c.ev_pct == 4.0  # (2000 / 50000) * 100 = 4.0
        assert c.base_win_rate == 0.55
        assert c.n == 100  # base_signals takes precedence
        assert c.backtest_period == "2023-01-01 to 2023-12-31"
        assert c.sharpe == 1.5
        assert c.advised_liquidity_pct == 10.0  # 0.10 * 100

    def test_basic_short_signal(self):
        """Single SHORT signal is converted to Candidate with direction='short'."""
        stats_rows = [
            {
                "strategy": "momentum",
                "symbol": "ETH",
                "direction": "SHORT",
                "entry": 3000.0,
                "stop": 3200.0,
                "target": 2800.0,
                "expected_value": 150.0,
                "base_sharpe": 0.8,
                "base_win_rate": 0.50,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.05,
            }
        ]
        advice_rows = [
            {
                "expected_value": 150.0,
                "entry": 3000.0,
                "base_win_rate": 0.50,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Sell ETH",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.direction == "short"
        assert c.ticker == "ETH"
        assert c.ev_pct == 5.0  # (150 / 3000) * 100 = 5.0
        assert c.n == 50

    def test_flat_signal_excluded(self):
        """FLAT direction rows are excluded entirely."""
        stats_rows = [
            {
                "strategy": "strategy1",
                "symbol": "BTC",
                "direction": "FLAT",
                "entry": 50000.0,
                "stop": 0.0,
                "target": 0.0,
                "expected_value": 0.0,
                "base_sharpe": 0.0,
                "base_win_rate": 0.5,
                "backtest_period": "period",
                "size": 0.0,
            }
        ]
        advice_rows = [
            {
                "expected_value": 0.0,
                "entry": 50000.0,
                "base_win_rate": 0.5,
                "base_signals": 10,
                "oracle_signals": None,
                "signal": "Exit",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)
        assert len(candidates) == 0

    def test_mixed_flat_and_directional(self):
        """Mix of FLAT and directional signals; only directional ones appear."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "A",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            },
            {
                "strategy": "s1",
                "symbol": "B",
                "direction": "FLAT",
                "entry": 200.0,
                "stop": 0.0,
                "target": 0.0,
                "expected_value": 0.0,
                "base_sharpe": 0.0,
                "base_win_rate": 0.5,
                "backtest_period": "p",
                "size": 0.0,
            },
            {
                "strategy": "s1",
                "symbol": "C",
                "direction": "SHORT",
                "entry": 150.0,
                "stop": 160.0,
                "target": 140.0,
                "expected_value": 7.5,
                "base_sharpe": 1.2,
                "base_win_rate": 0.52,
                "backtest_period": "p",
                "size": 0.08,
            },
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy A",
            },
            {
                "expected_value": 0.0,
                "entry": 200.0,
                "base_win_rate": 0.5,
                "base_signals": 10,
                "oracle_signals": None,
                "signal": "Exit B",
            },
            {
                "expected_value": 7.5,
                "entry": 150.0,
                "base_win_rate": 0.52,
                "base_signals": 30,
                "oracle_signals": None,
                "signal": "Sell C",
            },
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 2
        assert candidates[0].ticker == "A"
        assert candidates[0].direction == "long"
        assert candidates[1].ticker == "C"
        assert candidates[1].direction == "short"

    def test_fallback_oracle_signals(self):
        """When base_signals is missing, n falls back to oracle_signals."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": None,
                "oracle_signals": 200,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        assert candidates[0].n == 200

    def test_fallback_both_signals_missing(self):
        """When both base_signals and oracle_signals are missing, row is skipped."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": None,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 0

    def test_case_insensitive_direction(self):
        """Direction normalization handles case variations (uppercase -> lowercase)."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "long",  # lowercase
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        assert candidates[0].direction == "long"

    def test_advised_liquidity_pct_calculation(self):
        """advised_liquidity_pct is computed as size * 100."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.25,  # 25% of liquidity
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        assert candidates[0].advised_liquidity_pct == 25.0

    def test_multiple_signals_same_order(self):
        """Multiple signals maintain their order and proper pairing."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "A",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            },
            {
                "strategy": "s2",
                "symbol": "B",
                "direction": "SHORT",
                "entry": 200.0,
                "stop": 220.0,
                "target": 180.0,
                "expected_value": 10.0,
                "base_sharpe": 0.9,
                "base_win_rate": 0.50,
                "backtest_period": "p",
                "size": 0.15,
            },
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy A",
            },
            {
                "expected_value": 10.0,
                "entry": 200.0,
                "base_win_rate": 0.50,
                "base_signals": 75,
                "oracle_signals": None,
                "signal": "Sell B",
            },
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 2
        assert candidates[0].ticker == "A"
        assert candidates[0].strategy == "s1"
        assert candidates[0].n == 50
        assert candidates[1].ticker == "B"
        assert candidates[1].strategy == "s2"
        assert candidates[1].n == 75

    def test_ev_pct_positive_long_signal(self):
        """EV pct is correctly computed for LONG signals (positive expected_value)."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 1000.0,
                "stop": 950.0,
                "target": 1100.0,
                "expected_value": 100.0,  # 100 / 1000 * 100 = 10%
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 100.0,
                "entry": 1000.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert candidates[0].ev_pct == 10.0

    def test_ev_pct_negative_short_signal(self):
        """EV pct can be negative for SHORT signals (negative expected_value relative to entry)."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "SHORT",
                "entry": 1000.0,
                "stop": 1050.0,
                "target": 900.0,
                "expected_value": -50.0,  # -50 / 1000 * 100 = -5%
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": -50.0,
                "entry": 1000.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Sell TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert candidates[0].ev_pct == -5.0

    def test_missing_expected_value(self):
        """Row with missing expected_value is skipped (ev_pct would be None)."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": None,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": None,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 0

    def test_missing_entry(self):
        """Row with missing entry is skipped (ev_pct would be None)."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": None,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": None,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 0

    def test_zero_entry_skipped(self):
        """Row with zero entry is skipped (ev_pct computation guards against division by zero)."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 0.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 0.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 0

    def test_empty_input_lists(self):
        """Empty input lists return empty candidate list."""
        candidates = fetch_signals([], [])
        assert len(candidates) == 0

    def test_nullable_optional_fields_defaulting(self):
        """avg_win_pct, avg_loss_pct, avg_holding_days default to None."""
        stats_rows = [
            {
                "strategy": "s1",
                "symbol": "TEST",
                "direction": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "expected_value": 5.0,
                "base_sharpe": 1.0,
                "base_win_rate": 0.55,
                "backtest_period": "p",
                "size": 0.1,
            }
        ]
        advice_rows = [
            {
                "expected_value": 5.0,
                "entry": 100.0,
                "base_win_rate": 0.55,
                "base_signals": 50,
                "oracle_signals": None,
                "signal": "Buy TEST",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        assert candidates[0].avg_win_pct is None
        assert candidates[0].avg_loss_pct is None
        assert candidates[0].avg_holding_days is None


class TestValidateCandidate:
    """Test validate_candidate() schema validation function."""

    def test_valid_long_candidate(self):
        """Well-formed long candidate passes validation (returns None)."""
        c = Candidate(
            strategy="path_execution",
            ticker="REMX",
            direction="long",
            entry=50.0,
            stop=45.0,
            target=60.0,
            ev_pct=4.04,
            base_win_rate=0.47,
            n=161,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.23,
            advised_liquidity_pct=11.0,
        )
        assert validate_candidate(c) is None

    def test_valid_short_candidate(self):
        """Well-formed short candidate passes validation (returns None)."""
        c = Candidate(
            strategy="momentum",
            ticker="XYZ",
            direction="short",
            entry=100.0,
            stop=105.0,
            target=90.0,
            ev_pct=2.5,
            base_win_rate=0.52,
            n=200,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=0.95,
            advised_liquidity_pct=8.0,
        )
        assert validate_candidate(c) is None

    def test_missing_strategy_is_schema_error(self):
        """Candidate with None strategy returns SCHEMA_ERROR."""
        c = Candidate(
            strategy=None,
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_ticker_is_schema_error(self):
        """Candidate with None ticker returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker=None,
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_direction_is_schema_error(self):
        """Candidate with None direction returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction=None,
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_entry_is_schema_error(self):
        """Candidate with None entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=None,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_stop_is_schema_error(self):
        """Candidate with None stop returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=None,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_target_is_schema_error(self):
        """Candidate with None target returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=None,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_ev_pct_is_schema_error(self):
        """Candidate with None ev_pct returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=None,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_base_win_rate_is_schema_error(self):
        """Candidate with None base_win_rate returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=None,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_n_is_schema_error(self):
        """Candidate with None n returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=None,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_backtest_period_is_schema_error(self):
        """Candidate with None backtest_period returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period=None,
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_missing_sharpe_is_schema_error(self):
        """Candidate with None sharpe returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=None,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_nan_entry_is_schema_error(self):
        """Candidate with NaN entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=float("nan"),
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_nan_stop_is_schema_error(self):
        """Candidate with NaN stop returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=float("nan"),
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_nan_target_is_schema_error(self):
        """Candidate with NaN target returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=float("nan"),
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_nan_ev_pct_is_schema_error(self):
        """Candidate with NaN ev_pct returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=float("nan"),
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_nan_base_win_rate_is_schema_error(self):
        """Candidate with NaN base_win_rate returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=float("nan"),
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_nan_sharpe_is_schema_error(self):
        """Candidate with NaN sharpe returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=float("nan"),
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_inf_entry_is_schema_error(self):
        """Candidate with inf entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=float("inf"),
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_neg_inf_entry_is_schema_error(self):
        """Candidate with -inf entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=float("-inf"),
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_long_with_swapped_stop_target_is_schema_error(self):
        """Long candidate with stop > target (swapped) returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=110.0,  # Should be < entry
            target=95.0,  # Should be > entry
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_long_with_target_before_entry_is_schema_error(self):
        """Long candidate with target < entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=90.0,  # Should be > entry
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_long_with_stop_after_entry_is_schema_error(self):
        """Long candidate with stop > entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=105.0,  # Should be < entry
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_short_with_swapped_stop_target_is_schema_error(self):
        """Short candidate with target > stop (swapped) returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="short",
            entry=100.0,
            stop=95.0,  # Should be > entry
            target=105.0,  # Should be < entry
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_short_with_target_after_entry_is_schema_error(self):
        """Short candidate with target > entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="short",
            entry=100.0,
            stop=105.0,
            target=110.0,  # Should be < entry
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_short_with_stop_before_entry_is_schema_error(self):
        """Short candidate with stop < entry returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="short",
            entry=100.0,
            stop=95.0,  # Should be > entry
            target=90.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"

    def test_invalid_direction_is_schema_error(self):
        """Candidate with invalid direction (not 'long' or 'short') returns SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="invalid",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="period",
            sharpe=1.0,
            advised_liquidity_pct=10.0,
        )
        assert validate_candidate(c) == "SCHEMA_ERROR"


class TestAllocationConfig:
    """Test AllocationConfig dataclass and defaults."""

    def test_config_defaults(self):
        """AllocationConfig has all fields with RFC §3.1 defaults."""
        config = AllocationConfig()
        assert config.n0 == 100
        assert config.min_n == 50
        assert config.round_trip_cost_pct == 0.15
        assert config.kelly_mult == 0.35
        assert config.top_k == 12
        assert config.max_pos_pct == 15
        assert config.max_cluster_pct == 25
        assert config.gross_cap_pct == 100
        assert config.dust_min_pct == 1.0
        assert config.equity is None
        assert config.cluster_map == {}

    def test_config_custom_values(self):
        """AllocationConfig can be initialized with custom values."""
        config = AllocationConfig(
            n0=200,
            min_n=75,
            round_trip_cost_pct=0.20,
            kelly_mult=0.25,
            top_k=8,
            equity=100000.0,
            cluster_map={"BTC": "crypto", "AAPL": "tech"},
        )
        assert config.n0 == 200
        assert config.min_n == 75
        assert config.round_trip_cost_pct == 0.20
        assert config.kelly_mult == 0.25
        assert config.top_k == 8
        assert config.equity == 100000.0
        assert config.cluster_map == {"BTC": "crypto", "AAPL": "tech"}

    def test_config_cluster_map_independence(self):
        """Cluster map dicts are independent across instances."""
        config1 = AllocationConfig()
        config2 = AllocationConfig()
        config1.cluster_map["BTC"] = "crypto"
        assert "BTC" not in config2.cluster_map


class TestComputeDerived:
    """Test compute_derived() per-row formula implementation."""

    def test_compute_derived_geometry_fallback_path(self):
        """Geometry-fallback path (avg_win/avg_loss both None) computed correctly.

        Hand-computed example from RFC §4.2 worked example (NG=F close_direction):
        - risk=0.6, reward=6.7, b=11.2 (geometry), n=109, shrink=0.52
        - ev_pct=1.45%, ev_net=0.61% (after cost), Kelly=13.9%
        """
        c = Candidate(
            strategy="close_direction",
            ticker="NG=F",
            direction="long",
            entry=2.95,
            stop=2.97,
            target=3.14,
            ev_pct=1.45,
            base_win_rate=0.538,  # Approximate to match shrunk EV
            n=109,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=0.0,
            avg_win_pct=None,  # Fallback branch
            avg_loss_pct=None,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # Check risk and reward (geometry-based)
        assert abs(result["risk_pct"] - 0.677) < 0.01  # abs(2.97 - 2.95) / 2.95 * 100
        assert abs(result["reward_pct"] - 6.441) < 0.01  # abs(3.14 - 2.95) / 2.95 * 100

        # Check b (geometry fallback)
        assert abs(result["b"] - (result["reward_pct"] / result["risk_pct"])) < 0.001
        assert result["b"] > 0

        # Check shrink: 109 / (109 + 100) = 0.52147...
        assert abs(result["shrink"] - 0.521) < 0.001

        # Check ev_shrunk: 1.45 * 0.521 = 0.755
        assert abs(result["ev_shrunk"] - 0.755) < 0.01

        # Check ev_net: 0.755 - 0.15 = 0.605
        assert abs(result["ev_net"] - 0.605) < 0.01

        # Check kelly_frac (positive)
        assert result["kelly_frac"] > 0
        assert result["kelly_frac"] < 0.25  # Capped by kelly_mult=0.35

    def test_compute_derived_empirical_path(self):
        """Empirical path (avg_win/avg_loss both populated) uses actual payoff ratio.

        Hand-computed example from RFC §4.2 worked example (REMX):
        - risk=6.0, reward=7.6, n=161, shrink=0.617, ev_pct=4.04%
        - With avg_win=7.6, avg_loss=6.0, b=1.27
        """
        c = Candidate(
            strategy="path_execution",
            ticker="REMX",
            direction="short",
            entry=79.73,
            stop=84.51,
            target=73.71,
            ev_pct=4.04,
            base_win_rate=0.47,
            n=161,
            backtest_period="test",
            sharpe=1.23,
            advised_liquidity_pct=11.0,
            avg_win_pct=7.6,  # Empirical branch
            avg_loss_pct=6.0,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # Check risk and reward
        assert abs(result["risk_pct"] - 5.995) < 0.03  # abs(84.51 - 79.73) / 79.73 * 100
        assert abs(result["reward_pct"] - 7.533) < 0.03  # abs(73.71 - 79.73) / 79.73 * 100

        # Check b (empirical): 7.6 / 6.0 = 1.267
        assert abs(result["b"] - 1.267) < 0.01

        # Check loss_pct uses avg_loss_pct (not risk_pct)
        assert result["loss_pct"] == 6.0

        # Check shrink: 161 / (161 + 100) = 0.617
        assert abs(result["shrink"] - 0.617) < 0.001

        # Check ev_shrunk: 4.04 * 0.617 = 2.493
        assert abs(result["ev_shrunk"] - 2.493) < 0.01

        # Check ev_net: 2.493 - 0.15 = 2.343
        assert abs(result["ev_net"] - 2.343) < 0.01

        # Check score: ev_net / loss_pct = 2.343 / 6.0 = 0.39
        assert abs(result["score"] - 0.391) < 0.01

    def test_compute_derived_shrink_at_boundary_n_zero(self):
        """Shrink at boundary n=0 should be 0 (no confidence in zero trades)."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=0,  # Boundary: no trades
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # shrink = 0 / (0 + 100) = 0
        assert result["shrink"] == 0.0
        # ev_shrunk = 5.0 * 0 = 0
        assert result["ev_shrunk"] == 0.0
        # ev_net = 0 - 0.15 = -0.15
        assert result["ev_net"] == -0.15
        # p_shrunk = 0.5 + (0.55 - 0.5) * 0 = 0.5 (no update)
        assert result["p_shrunk"] == 0.5
        # With b=reward/risk=10/5=2, kelly_raw = 0.5 - 0.5/2 = 0.25
        # kelly_frac = 0.25 * 0.35 = 0.0875 (positive, good ratio)
        assert abs(result["kelly_frac"] - 0.0875) < 0.001

    def test_compute_derived_shrink_at_large_n(self):
        """Shrink at large n approaches 1 (full confidence in many trades)."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=10000,  # Large: many trades
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # shrink = 10000 / (10000 + 100) ≈ 0.9901
        assert abs(result["shrink"] - 0.9901) < 0.001
        assert result["shrink"] < 1.0  # Never quite reaches 1
        # ev_shrunk approaches ev_pct
        assert abs(result["ev_shrunk"] - (5.0 * result["shrink"])) < 0.001
        # p_shrunk approaches base_win_rate
        assert abs(result["p_shrunk"] - 0.55) < 0.001

    def test_compute_derived_no_division_by_zero(self):
        """All divisions are safe; denominators are never zero."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=50,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
            avg_win_pct=None,
            avg_loss_pct=None,
        )
        config = AllocationConfig()

        # Should not raise ZeroDivisionError
        result = compute_derived(c, config)

        # All results should be finite
        for key, value in result.items():
            assert math.isfinite(value), f"{key}={value} is not finite"

    def test_compute_derived_kelly_raw_can_be_negative(self):
        """Kelly raw can be negative; kelly_frac clamps to 0."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward (2:1 ratio)
            ev_pct=-10.0,  # Negative edge, but geometry is still 2:1
            base_win_rate=0.1,  # Very low win rate (10%)
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # With shrink = 100/(100+100) = 0.5
        # p_shrunk = 0.5 + (0.1 - 0.5) * 0.5 = 0.5 - 0.2 = 0.3
        # b = 10/5 = 2
        # kelly_raw = 0.3 - (1 - 0.3) / 2 = 0.3 - 0.35 = -0.05
        assert result["kelly_raw"] < 0
        # kelly_frac should be 0 (clamped by max)
        assert result["kelly_frac"] == 0.0

    def test_compute_derived_consistent_with_worked_example_v_stock(self):
        """Test with V (Visa) stock from RFC §4.2 worked example.

        V: risk=1.6%, reward=1.8%, n=319, shrink=0.761, ev_pct=0.68%
        """
        c = Candidate(
            strategy="strategy1",
            ticker="V",
            direction="long",
            entry=200.0,
            stop=196.8,  # 1.6% below entry
            target=203.6,  # 1.8% above entry
            ev_pct=0.68,
            base_win_rate=0.52,
            n=319,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
            avg_win_pct=None,
            avg_loss_pct=None,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # Verify shrink: 319 / (319 + 100) ≈ 0.761
        assert abs(result["shrink"] - 0.761) < 0.001

        # Verify risk/reward geometry
        assert abs(result["risk_pct"] - 1.6) < 0.01
        assert abs(result["reward_pct"] - 1.8) < 0.01
        assert abs(result["b"] - 1.125) < 0.01

    def test_compute_derived_output_keys(self):
        """compute_derived returns dict with all required keys."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        required_keys = {
            "risk_pct", "reward_pct", "b", "loss_pct", "shrink", "ev_shrunk",
            "ev_net", "p_shrunk", "kelly_raw", "kelly_frac", "score"
        }
        assert set(result.keys()) == required_keys

    def test_compute_derived_with_custom_config(self):
        """compute_derived respects custom AllocationConfig values."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig(
            n0=200,  # Custom shrinkage constant
            round_trip_cost_pct=0.20,  # Custom cost
            kelly_mult=0.25,  # Custom kelly multiplier
        )
        result = compute_derived(c, config)

        # shrink should use custom n0: 100 / (100 + 200) = 0.333...
        assert abs(result["shrink"] - 0.333) < 0.01

        # ev_net should use custom cost
        expected_ev_net = 5.0 * result["shrink"] - 0.20
        assert abs(result["ev_net"] - expected_ev_net) < 0.001

        # kelly_frac should use custom multiplier
        assert result["kelly_frac"] < 0.25  # Capped by kelly_mult

    def test_compute_derived_short_position(self):
        """compute_derived handles short positions correctly (abs values)."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="short",
            entry=100.0,
            stop=105.0,  # Higher than entry (stop for short)
            target=90.0,  # Lower than entry (target for short)
            ev_pct=3.5,
            base_win_rate=0.52,
            n=80,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # risk_pct = abs(105 - 100) / 100 * 100 = 5
        assert abs(result["risk_pct"] - 5.0) < 0.01
        # reward_pct = abs(90 - 100) / 100 * 100 = 10
        assert abs(result["reward_pct"] - 10.0) < 0.01
        # b = 10 / 5 = 2
        assert abs(result["b"] - 2.0) < 0.01

    def test_compute_derived_empirical_only_avg_win(self):
        """When only avg_win_pct is present, fallback to geometry."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
            avg_win_pct=5.5,  # Provided
            avg_loss_pct=None,  # Missing
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # Should use geometry fallback (both not present)
        assert abs(result["b"] - (result["reward_pct"] / result["risk_pct"])) < 0.001
        assert result["loss_pct"] == result["risk_pct"]

    def test_compute_derived_empirical_only_avg_loss(self):
        """When only avg_loss_pct is present, fallback to geometry."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
            avg_win_pct=None,  # Missing
            avg_loss_pct=4.5,  # Provided
        )
        config = AllocationConfig()
        result = compute_derived(c, config)

        # Should use geometry fallback (both not present)
        assert abs(result["b"] - (result["reward_pct"] / result["risk_pct"])) < 0.001
        assert result["loss_pct"] == result["risk_pct"]


class TestComputeEvRatio:
    """Test compute_ev_ratio() data-quality check per RFC §4.3."""

    def test_ev_ratio_inside_band_no_mismatch(self):
        """EV ratio inside [0.5, 2.0] band is not mismatched."""
        c = Candidate(
            strategy="path_execution",
            ticker="REMX",
            direction="short",
            entry=79.73,
            stop=84.51,
            target=73.71,
            ev_pct=4.04,
            base_win_rate=0.47,
            n=161,
            backtest_period="test",
            sharpe=1.23,
            advised_liquidity_pct=11.0,
        )
        # Compute derived to get risk_pct and reward_pct
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.47 * 7.533 - (1 - 0.47) * 5.995
        #            = 0.47 * 7.533 - 0.53 * 5.995
        #            = 3.540 - 3.177
        #            = 0.363
        # ev_ratio = 4.04 / 0.363 ≈ 11.1 (outside band, so this is actually a mismatch)
        # Let's instead construct a case where ev_pct matches geometry well
        c2 = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=2.5,  # Conservative estimate, inside band
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        derived2 = compute_derived(c2, config)

        # ev_implied = 0.55 * 10.0 - 0.45 * 5.0
        #            = 5.5 - 2.25
        #            = 3.25
        # ev_ratio = 2.5 / 3.25 ≈ 0.769 (inside [0.5, 2.0])
        ev_ratio, is_mismatch = compute_ev_ratio(c2, derived2)

        assert abs(ev_ratio - 0.769) < 0.01
        assert is_mismatch is False

    def test_ev_ratio_above_2_0_is_mismatch(self):
        """EV ratio > 2.0 is flagged as mismatched."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=7.0,  # Much higher than geometry implies
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.55 * 10.0 - 0.45 * 5.0
        #            = 5.5 - 2.25
        #            = 3.25
        # ev_ratio = 7.0 / 3.25 ≈ 2.15 (above 2.0)
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)

        assert ev_ratio > 2.0
        assert is_mismatch is True

    def test_ev_ratio_below_0_5_is_mismatch(self):
        """EV ratio < 0.5 is flagged as mismatched."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=1.0,  # Much lower than geometry implies
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.55 * 10.0 - 0.45 * 5.0
        #            = 5.5 - 2.25
        #            = 3.25
        # ev_ratio = 1.0 / 3.25 ≈ 0.308 (below 0.5)
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)

        assert ev_ratio < 0.5
        assert is_mismatch is True

    def test_ev_ratio_near_zero_ev_implied_no_mismatch(self):
        """When ev_implied is near zero, ratio is undefined; treat as not-mismatched."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=10.0,  # Arbitrary positive EV
            base_win_rate=0.5,  # 50% win rate
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.5 * 10.0 - 0.5 * 5.0
        #            = 5.0 - 2.5
        #            = 2.5 (not near zero)
        # Let's pick a case where ev_implied is actually near zero
        c2 = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=99.0,  # 1% risk
            target=101.0,  # 1% reward (symmetric, exactly fair at 50%)
            ev_pct=100.0,  # Arbitrarily large but ev_implied will be ~0
            base_win_rate=0.5,  # Exactly 50%
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        derived2 = compute_derived(c2, config)

        # ev_implied = 0.5 * 1.0 - 0.5 * 1.0
        #            = 0.5 - 0.5
        #            = 0.0 (near zero guard triggers)
        ev_ratio, is_mismatch = compute_ev_ratio(c2, derived2)

        assert ev_ratio == 0.0
        assert is_mismatch is False

    def test_ev_ratio_boundary_0_5_exact(self):
        """EV ratio exactly at 0.5 (lower boundary) is not mismatched."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=1.625,  # Computed to give exactly 0.5 ratio
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.55 * 10.0 - 0.45 * 5.0 = 3.25
        # ev_ratio = 1.625 / 3.25 = 0.5 (exactly at boundary)
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)

        assert abs(ev_ratio - 0.5) < 1e-6
        assert is_mismatch is False

    def test_ev_ratio_boundary_2_0_exact(self):
        """EV ratio exactly at 2.0 (upper boundary) is not mismatched."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=6.5,  # Computed to give exactly 2.0 ratio
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.55 * 10.0 - 0.45 * 5.0 = 3.25
        # ev_ratio = 6.5 / 3.25 = 2.0 (exactly at boundary)
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)

        assert abs(ev_ratio - 2.0) < 1e-6
        assert is_mismatch is False

    def test_ev_ratio_negative_ev_pct(self):
        """Negative ev_pct (edge case) is handled correctly."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=-2.0,  # Negative EV
            base_win_rate=0.55,
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.55 * 10.0 - 0.45 * 5.0 = 3.25
        # ev_ratio = -2.0 / 3.25 ≈ -0.615 (negative ratio)
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)

        assert ev_ratio < 0
        assert is_mismatch is True  # Outside [0.5, 2.0] band

    def test_ev_ratio_negative_ev_implied(self):
        """Negative ev_implied (low win rate, unfavorable geometry)."""
        c = Candidate(
            strategy="test",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=102.0,  # 2% reward (bad ratio for 30% win rate)
            ev_pct=1.0,
            base_win_rate=0.30,  # Low win rate
            n=100,
            backtest_period="test",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        derived = compute_derived(c, config)

        # ev_implied = 0.30 * 2.0 - 0.70 * 5.0
        #            = 0.6 - 3.5
        #            = -2.9 (negative, terrible geometry)
        # ev_ratio = 1.0 / -2.9 ≈ -0.345 (negative ratio)
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)

        assert ev_ratio < 0
        assert is_mismatch is True


class TestSelectCandidates:
    """Test select_candidates() gating, collapse, and top-K ranking.

    Per ticket E11-S05, implements RFC §4.4 selection with these stages:
    1. Gate: SCHEMA_ERROR → DISABLED → LOW_N → NEG_EV_NET
    2. Per-asset collapse: DIRECTION_CONFLICT or DUP_ASSET
    3. Rank + top-K: BELOW_TOPK or proceed to E11-S06
    """

    def test_gate_schema_error_invalid_direction(self):
        """Candidate with invalid direction rejected as SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="invalid",  # Not "long" or "short"
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] == "SCHEMA_ERROR"
        assert result[0]["flags"] == []
        assert result[0]["derived"] == {}

    def test_gate_schema_error_long_bad_placement(self):
        """Long with stop >= entry rejected as SCHEMA_ERROR."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=105.0,  # Should be < entry for long
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] == "SCHEMA_ERROR"

    def test_gate_disabled(self):
        """Candidate with enabled_mask[ticker]=False rejected as DISABLED."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        enabled_mask = {"TEST": False}
        result = select_candidates([c], config, enabled_mask)

        assert len(result) == 1
        assert result[0]["status"] == "DISABLED"
        assert result[0]["flags"] == []
        assert result[0]["derived"]["score"] is not None

    def test_gate_low_n(self):
        """Candidate with n < config.min_n rejected as LOW_N."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=25,  # Less than default min_n=50
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig(min_n=50)
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] == "LOW_N"

    def test_gate_neg_ev_net(self):
        """Candidate with ev_net <= 0 rejected as NEG_EV_NET."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=0.1,  # Very low; will be shrunk below cost
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig(round_trip_cost_pct=0.15)
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] == "NEG_EV_NET"
        # Verify ev_net is indeed <= 0
        assert result[0]["derived"]["ev_net"] <= 0

    def test_gate_order_schema_error_before_disabled(self):
        """SCHEMA_ERROR is caught before DISABLED in gate order."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="invalid",  # SCHEMA_ERROR
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        enabled_mask = {"TEST": False}  # Would trigger DISABLED if checked
        result = select_candidates([c], config, enabled_mask)

        # Should be SCHEMA_ERROR, not DISABLED (earlier in order)
        assert result[0]["status"] == "SCHEMA_ERROR"

    def test_gate_order_low_n_before_neg_ev_net(self):
        """LOW_N is caught before NEG_EV_NET in gate order."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=0.1,  # Would trigger NEG_EV_NET if checked
            base_win_rate=0.55,
            n=25,  # LOW_N
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig(min_n=50)
        result = select_candidates([c], config, {})

        # Should be LOW_N, not NEG_EV_NET
        assert result[0]["status"] == "LOW_N"

    def test_gate_survivor_status_none(self):
        """Candidate passing all gates has status=None."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] is None
        assert result[0]["derived"]["score"] is not None

    def test_data_mismatch_flag_outside_band(self):
        """Candidate with ev_ratio outside [0.5, 2.0] gets DATA_MISMATCH flag."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=7.0,  # Much higher than geometry implies (ratio > 2.0)
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] is None
        assert result[0]["flags"] == ["DATA_MISMATCH"]

    def test_data_mismatch_flag_inside_band(self):
        """Candidate with ev_ratio inside [0.5, 2.0] has empty flags."""
        c = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,  # 5% risk
            target=110.0,  # 10% reward
            ev_pct=2.5,  # Reasonable, inside band
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c], config, {})

        assert len(result) == 1
        assert result[0]["status"] is None
        assert result[0]["flags"] == []

    def test_direction_conflict_both_long_short(self):
        """Two candidates same ticker (long + short) both gated as DIRECTION_CONFLICT."""
        c1 = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        c2 = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="short",
            entry=100.0,
            stop=105.0,
            target=90.0,
            ev_pct=4.0,
            base_win_rate=0.52,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c1, c2], config, {})

        assert len(result) == 2
        assert result[0]["status"] == "DIRECTION_CONFLICT"
        assert result[1]["status"] == "DIRECTION_CONFLICT"

    def test_dup_asset_keeps_higher_score(self):
        """Two candidates same ticker/direction; higher-score kept, other rejected DUP_ASSET."""
        c1 = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,  # Higher EV, should have higher score
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        c2 = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=2.0,  # Lower EV, should have lower score
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c1, c2], config, {})

        assert len(result) == 2
        # First candidate should survive (higher score)
        assert result[0]["status"] is None
        assert result[0]["ticker"] == "TEST"
        # Second candidate should be rejected as DUP_ASSET
        assert result[1]["status"] == "DUP_ASSET"
        assert result[1]["ticker"] == "TEST"

    def test_dup_asset_preserves_higher_score_row(self):
        """When collapsing duplicates, the higher-score row survives."""
        c1 = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=2.0,  # Lower EV initially
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        c2 = Candidate(
            strategy="s1",
            ticker="TEST",
            direction="long",
            entry=100.0,
            stop=95.0,
            target=110.0,
            ev_pct=5.0,  # Higher EV
            base_win_rate=0.55,
            n=100,
            backtest_period="p",
            sharpe=1.0,
            advised_liquidity_pct=5.0,
        )
        config = AllocationConfig()
        result = select_candidates([c1, c2], config, {})

        assert len(result) == 2
        # Second row (c2 with higher score) should survive
        assert result[1]["status"] is None
        assert result[0]["status"] == "DUP_ASSET"

    def test_below_topk_exceeding_top_k_limit(self):
        """Survivors beyond config.top_k are rejected as BELOW_TOPK."""
        candidates = []
        for i in range(15):
            c = Candidate(
                strategy="s1",
                ticker=f"TEST{i}",  # Unique ticker
                direction="long",
                entry=100.0,
                stop=95.0,
                target=110.0,
                ev_pct=5.0 - i * 0.2,  # Descending EV for ranking
                base_win_rate=0.55,
                n=100,
                backtest_period="p",
                sharpe=1.0,
                advised_liquidity_pct=5.0,
            )
            candidates.append(c)

        config = AllocationConfig(top_k=12)
        result = select_candidates(candidates, config, {})

        assert len(result) == 15
        # First 12 should have status=None
        for i in range(12):
            assert result[i]["status"] is None, f"Row {i} should have status=None"
        # Remaining 3 should have status=BELOW_TOPK
        for i in range(12, 15):
            assert result[i]["status"] == "BELOW_TOPK", f"Row {i} should have status=BELOW_TOPK"

    def test_ranking_by_score_descending(self):
        """Survivors ranked by score descending; top config.top_k selected."""
        candidates = []
        # Create candidates with varying scores (lower index = higher score)
        for i in range(5):
            c = Candidate(
                strategy="s1",
                ticker=f"T{i}",
                direction="long",
                entry=100.0,
                stop=95.0,
                target=110.0,
                ev_pct=4.0 - i * 0.8,  # i=0 has highest EV
                base_win_rate=0.55,
                n=100,
                backtest_period="p",
                sharpe=1.0,
                advised_liquidity_pct=5.0,
            )
            candidates.append(c)

        config = AllocationConfig(top_k=3)
        result = select_candidates(candidates, config, {})

        # Collect survivors and their scores in result order
        survivors = []
        for r in result:
            if r["status"] is None or r["status"] == "BELOW_TOPK":
                survivors.append((r["ticker"], r["derived"]["score"], r["status"]))

        # Verify top 3 have status=None, rest BELOW_TOPK
        assert survivors[0][2] is None  # Highest score (T0)
        assert survivors[1][2] is None  # Second highest (T1)
        assert survivors[2][2] is None  # Third highest (T2)
        assert survivors[3][2] == "BELOW_TOPK"  # Below top-K (T3)
        assert survivors[4][2] == "BELOW_TOPK"  # Below top-K (T4)

    def test_happy_path_multi_ticker_scenario(self):
        """Multi-ticker scenario: some pass, some rejected, top-K selected."""
        # Create a mix of candidates
        candidates = [
            # Ticker A: long (good)
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
            # Ticker B: short (good, higher score)
            Candidate("s1", "B", "short", 100.0, 105.0, 90.0, 6.0, 0.55, 100, "p", 1.0, 5.0),
            # Ticker C: low ev_net (rejected)
            Candidate("s1", "C", "long", 100.0, 95.0, 110.0, 0.1, 0.55, 100, "p", 1.0, 5.0),
            # Ticker D: long (good)
            Candidate("s1", "D", "long", 100.0, 95.0, 110.0, 4.0, 0.55, 100, "p", 1.0, 5.0),
            # Ticker A: short (direction conflict with long A)
            Candidate("s1", "A", "short", 100.0, 105.0, 90.0, 3.0, 0.55, 100, "p", 1.0, 5.0),
        ]

        config = AllocationConfig(top_k=2)
        result = select_candidates(candidates, config, {})

        assert len(result) == 5

        # Ticker C should be NEG_EV_NET
        c_result = result[2]
        assert c_result["ticker"] == "C"
        assert c_result["status"] == "NEG_EV_NET"

        # Ticker A: both long and short, so DIRECTION_CONFLICT for both
        a_long_result = result[0]
        a_short_result = result[4]
        assert a_long_result["ticker"] == "A"
        assert a_long_result["status"] == "DIRECTION_CONFLICT"
        assert a_short_result["ticker"] == "A"
        assert a_short_result["status"] == "DIRECTION_CONFLICT"

        # Ticker B and D should pass gating/collapse
        b_result = result[1]
        d_result = result[3]
        assert b_result["ticker"] == "B"
        assert b_result["status"] is None  # In top-K
        assert d_result["ticker"] == "D"
        assert d_result["status"] is None  # In top-K (both B and D in top-K)

    def test_output_returns_all_candidates(self):
        """Output includes all candidates (rejected and selected)."""
        c1 = Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0)
        c2 = Candidate("s1", "B", "long", 100.0, 95.0, 110.0, 0.1, 0.55, 100, "p", 1.0, 5.0)
        config = AllocationConfig()
        result = select_candidates([c1, c2], config, {})

        assert len(result) == 2
        # First should survive
        assert result[0]["ticker"] == "A"
        assert result[0]["status"] is None
        # Second should be rejected
        assert result[1]["ticker"] == "B"
        assert result[1]["status"] == "NEG_EV_NET"

    def test_output_preserves_input_order(self):
        """Output order matches input order (not sorted by score)."""
        candidates = []
        for i in range(5):
            c = Candidate(
                "s1", f"T{i}", "long", 100.0, 95.0, 110.0,
                5.0 - i * 0.5,  # Descending scores
                0.55, 100, "p", 1.0, 5.0
            )
            candidates.append(c)

        config = AllocationConfig(top_k=5)
        result = select_candidates(candidates, config, {})

        # Verify output is in input order, not sorted by score
        for i in range(5):
            assert result[i]["ticker"] == f"T{i}"

    def test_output_row_contains_all_fields(self):
        """Each output row contains all required fields."""
        c = Candidate("s1", "TEST", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0)
        config = AllocationConfig()
        result = select_candidates([c], config, {})

        row = result[0]
        # Original Candidate fields
        assert "strategy" in row
        assert "ticker" in row
        assert "direction" in row
        assert "entry" in row
        assert "stop" in row
        assert "target" in row
        assert "ev_pct" in row
        assert "base_win_rate" in row
        assert "n" in row
        assert "backtest_period" in row
        assert "sharpe" in row
        assert "advised_liquidity_pct" in row
        assert "avg_win_pct" in row
        assert "avg_loss_pct" in row
        assert "avg_holding_days" in row
        # New fields
        assert "derived" in row
        assert "status" in row
        assert "flags" in row
        # Derived should have score key
        assert "score" in row["derived"]

    def test_enabled_mask_defaults_to_enabled(self):
        """Tickers not in enabled_mask default to enabled."""
        c = Candidate("s1", "TEST", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0)
        config = AllocationConfig()
        # Empty enabled_mask
        result = select_candidates([c], config, {})

        # Should pass (default enabled)
        assert result[0]["status"] is None

    def test_enabled_mask_true_allows_candidate(self):
        """enabled_mask[ticker]=True allows candidate."""
        c = Candidate("s1", "TEST", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0)
        config = AllocationConfig()
        enabled_mask = {"TEST": True}
        result = select_candidates([c], config, enabled_mask)

        # Should pass
        assert result[0]["status"] is None

    def test_empty_candidate_list(self):
        """Empty candidate list returns empty result."""
        config = AllocationConfig()
        result = select_candidates([], config, {})

        assert result == []

    def test_all_candidates_rejected_in_gate(self):
        """All candidates rejected in gating stage."""
        candidates = [
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 0.1, 0.55, 100, "p", 1.0, 5.0),
            Candidate("s1", "B", "long", 100.0, 95.0, 110.0, 0.1, 0.55, 100, "p", 1.0, 5.0),
        ]
        config = AllocationConfig()
        result = select_candidates(candidates, config, {})

        assert len(result) == 2
        assert all(r["status"] == "NEG_EV_NET" for r in result)

    def test_collapse_reduces_candidate_count_after_selection(self):
        """After collapse, duplicate ticker with lower score is rejected."""
        # Three candidates: A (good), B (good), A_dup (duplicate A, lower score)
        candidates = [
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
            Candidate("s1", "B", "long", 100.0, 95.0, 110.0, 4.0, 0.55, 100, "p", 1.0, 5.0),
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 2.0, 0.55, 100, "p", 1.0, 5.0),
        ]
        config = AllocationConfig(top_k=2)
        result = select_candidates(candidates, config, {})

        assert len(result) == 3
        # A first should survive (higher score)
        assert result[0]["ticker"] == "A"
        assert result[0]["status"] is None
        # B should survive (in top-K)
        assert result[1]["ticker"] == "B"
        assert result[1]["status"] is None
        # A duplicate should be DUP_ASSET
        assert result[2]["ticker"] == "A"
        assert result[2]["status"] == "DUP_ASSET"


class TestSizeSelected:
    """Test size_selected() sizing pipeline (position cap → cluster caps → gross cap → dust)."""

    def _make_survivor(self, ticker, kelly_frac, cluster="default", ev_net=2.0, status=None):
        """Helper to create a survivor row with minimal fields."""
        return {
            "ticker": ticker,
            "strategy": "s1",
            "direction": "long",
            "entry": 100.0,
            "stop": 95.0,
            "target": 110.0,
            "ev_pct": 5.0,
            "base_win_rate": 0.55,
            "n": 100,
            "backtest_period": "p",
            "sharpe": 1.0,
            "advised_liquidity_pct": 5.0,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "avg_holding_days": None,
            "derived": {
                "kelly_frac": kelly_frac,
                "risk_pct": 5.0,
                "reward_pct": 10.0,
                "b": 2.0,
                "loss_pct": 5.0,
                "shrink": 0.5,
                "ev_shrunk": 2.5,
                "ev_net": ev_net,
                "p_shrunk": 0.55,
                "kelly_raw": 0.2,
                "score": 0.4,
            },
            "status": status,
            "flags": [],
        }

    def test_happy_path_no_caps_triggered(self):
        """No caps triggered; all survivors remain as-is with status=SELECTED."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.10),  # 10% alloc, no cap
            self._make_survivor("B", kelly_frac=0.08),  # 8% alloc, no cap
        ]
        config = AllocationConfig(max_pos_pct=15.0, max_cluster_pct=25.0, gross_cap_pct=100.0)
        config.cluster_map = {}

        result = size_selected(survivors, config)

        assert len(result) == 2
        # Both should have been sized
        assert result[0]["alloc"] == 10.0  # kelly_frac * 100
        assert result[0]["status"] == "SELECTED"
        assert result[0]["flags"] == []

        assert result[1]["alloc"] == 8.0
        assert result[1]["status"] == "SELECTED"
        assert result[1]["flags"] == []

    def test_position_cap_triggered(self):
        """Position cap triggered; kelly_frac * 100 clamped to max_pos_pct."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.20),  # 20% kelly, capped to 15%
        ]
        config = AllocationConfig(max_pos_pct=15.0, max_cluster_pct=25.0, gross_cap_pct=100.0)
        config.cluster_map = {}

        result = size_selected(survivors, config)

        assert len(result) == 1
        assert result[0]["alloc"] == 15.0  # Capped to max_pos_pct
        assert result[0]["status"] == "SELECTED"
        assert "POS_CAPPED" in result[0]["flags"]

    def test_cluster_cap_triggered(self):
        """Cluster cap triggered; cluster allocations scaled proportionally."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.10, cluster="tech"),  # 10%
            self._make_survivor("B", kelly_frac=0.12, cluster="tech"),  # 12%
            self._make_survivor("C", kelly_frac=0.08, cluster="energy"),  # 8%
        ]
        # tech cluster: 10 + 12 = 22% > 20% cap, needs scaling
        config = AllocationConfig(
            max_pos_pct=15.0, max_cluster_pct=20.0, gross_cap_pct=100.0
        )
        config.cluster_map = {"A": "tech", "B": "tech", "C": "energy"}

        result = size_selected(survivors, config)

        assert len(result) == 3
        # Tech cluster sum before: 10 + 12 = 22
        # Scale factor: 20 / 22 = 0.909...
        # A: 10 * 0.909 = 9.09, B: 12 * 0.909 = 10.909, C: 8 (unchanged)
        assert abs(result[0]["alloc"] - (10.0 * 20.0 / 22.0)) < 0.01
        assert abs(result[1]["alloc"] - (12.0 * 20.0 / 22.0)) < 0.01
        assert abs(result[2]["alloc"] - 8.0) < 0.01

        # Tech positions should have CLUSTER_CAPPED flag
        assert "CLUSTER_CAPPED" in result[0]["flags"]
        assert "CLUSTER_CAPPED" in result[1]["flags"]
        assert "CLUSTER_CAPPED" not in result[2]["flags"]

        # All should be SELECTED
        for r in result:
            assert r["status"] == "SELECTED"

    def test_gross_cap_triggered(self):
        """Gross cap triggered; all allocations scaled by single factor."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.08),  # 8%
            self._make_survivor("B", kelly_frac=0.07),  # 7%
            self._make_survivor("C", kelly_frac=0.06),  # 6%
        ]
        # Total: 8 + 7 + 6 = 21% > 20% cap
        config = AllocationConfig(
            max_pos_pct=15.0, max_cluster_pct=25.0, gross_cap_pct=20.0
        )
        config.cluster_map = {}

        result = size_selected(survivors, config)

        assert len(result) == 3
        # Scale factor: 20 / 21 = 0.952...
        scale_factor = 20.0 / 21.0
        assert abs(result[0]["alloc"] - (8.0 * scale_factor)) < 0.01
        assert abs(result[1]["alloc"] - (7.0 * scale_factor)) < 0.01
        assert abs(result[2]["alloc"] - (6.0 * scale_factor)) < 0.01

        # All should be SELECTED
        for r in result:
            assert r["status"] == "SELECTED"

    def test_dust_filter_zeroes_small_allocation(self):
        """Dust filter; allocation < dust_min_pct zeroed and marked DUST."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.05),  # 5%
            self._make_survivor("B", kelly_frac=0.003),  # 0.3%, below 1% dust threshold
        ]
        config = AllocationConfig(
            max_pos_pct=15.0, max_cluster_pct=25.0, gross_cap_pct=100.0, dust_min_pct=1.0
        )
        config.cluster_map = {}

        result = size_selected(survivors, config)

        assert len(result) == 2
        assert result[0]["alloc"] == 5.0
        assert result[0]["status"] == "SELECTED"

        assert result[1]["alloc"] == 0.0  # Zeroed
        assert result[1]["status"] == "DUST"

    def test_rejected_rows_pass_through_unchanged(self):
        """Rejected rows (status != None) pass through unchanged."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.10, status=None),  # Top-K
            {
                **self._make_survivor("B", kelly_frac=0.05),
                "status": "BELOW_TOPK",  # Rejected
            },
            {
                **self._make_survivor("C", kelly_frac=0.07),
                "status": "LOW_N",  # Rejected
            },
        ]
        config = AllocationConfig(max_pos_pct=15.0, max_cluster_pct=25.0, gross_cap_pct=100.0)
        config.cluster_map = {}

        result = size_selected(survivors, config)

        assert len(result) == 3
        # A should be sized
        assert result[0]["status"] == "SELECTED"
        assert result[0]["alloc"] == 10.0

        # B and C should be unchanged
        assert result[1]["status"] == "BELOW_TOPK"
        assert "alloc" not in result[1]  # No alloc field added to rejected rows

        assert result[2]["status"] == "LOW_N"
        assert "alloc" not in result[2]

    def test_combined_cluster_gross_dust_cascade(self):
        """Combined scenario: cluster cap → gross cap → dust in sequence."""
        # Setup: 3 clusters, each with 2 positions
        # tech: A (10%), B (12%) -> cluster sum 22% > 20% cluster cap
        # energy: C (8%), D (9%) -> cluster sum 17% < 20% cluster cap
        # healthcare: E (0.5%), F (1%) -> cluster sum 1.5% < 20% cluster cap
        # After cluster caps: tech scaled to 20%, energy stays 17%, healthcare stays 1.5%
        # Total after cluster: 38.5% > 30% gross cap -> scale by 30/38.5
        # E: 0.5 * scale < 1.0% dust threshold -> zeroed to DUST
        # F: 1 * scale may still be < 1.0% depending on scale -> check
        survivors = [
            self._make_survivor("A", kelly_frac=0.10, cluster="tech"),
            self._make_survivor("B", kelly_frac=0.12, cluster="tech"),
            self._make_survivor("C", kelly_frac=0.08, cluster="energy"),
            self._make_survivor("D", kelly_frac=0.09, cluster="energy"),
            self._make_survivor("E", kelly_frac=0.005, cluster="healthcare"),
            self._make_survivor("F", kelly_frac=0.01, cluster="healthcare"),
        ]
        config = AllocationConfig(
            max_pos_pct=15.0,
            max_cluster_pct=20.0,
            gross_cap_pct=30.0,
            dust_min_pct=1.0,
        )
        config.cluster_map = {
            "A": "tech",
            "B": "tech",
            "C": "energy",
            "D": "energy",
            "E": "healthcare",
            "F": "healthcare",
        }

        result = size_selected(survivors, config)

        assert len(result) == 6

        # After cluster cap, before gross cap:
        # Tech: (10 + 12) * (20/22) = 20
        # Energy: 8 + 9 = 17 (unchanged)
        # Healthcare: 0.5 + 1 = 1.5 (unchanged)
        # Total: 38.5

        # After gross cap: scale by 30/38.5 = 0.7792
        gross_scale = 30.0 / 38.5
        cluster_tech_scale = 20.0 / 22.0

        # A, B: tech cluster scaled, then gross scaled
        a_expected = 10.0 * cluster_tech_scale * gross_scale
        b_expected = 12.0 * cluster_tech_scale * gross_scale
        c_expected = 8.0 * gross_scale
        d_expected = 9.0 * gross_scale
        e_expected = 0.5 * gross_scale  # Likely < 1.0 dust threshold
        f_expected = 1.0 * gross_scale  # Likely < 1.0 dust threshold

        assert abs(result[0]["alloc"] - a_expected) < 0.01
        assert abs(result[1]["alloc"] - b_expected) < 0.01
        assert abs(result[2]["alloc"] - c_expected) < 0.01
        assert abs(result[3]["alloc"] - d_expected) < 0.01

        # Check dust filter results
        if e_expected < config.dust_min_pct:
            assert result[4]["alloc"] == 0.0
            assert result[4]["status"] == "DUST"
        else:
            assert result[4]["status"] == "SELECTED"

        if f_expected < config.dust_min_pct:
            assert result[5]["alloc"] == 0.0
            assert result[5]["status"] == "DUST"
        else:
            assert result[5]["status"] == "SELECTED"

        # C and D should be SELECTED (above dust)
        assert result[2]["status"] == "SELECTED"
        assert result[3]["status"] == "SELECTED"

    def test_position_cap_and_cluster_cap_combined(self):
        """Position cap clamping happens before cluster cap scaling."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.20, cluster="tech"),  # Clamped to 15%
            self._make_survivor("B", kelly_frac=0.12, cluster="tech"),  # 12%
        ]
        # Tech cluster after pos cap: 15% + 12% = 27% > 20% cluster cap
        config = AllocationConfig(
            max_pos_pct=15.0, max_cluster_pct=20.0, gross_cap_pct=100.0
        )
        config.cluster_map = {"A": "tech", "B": "tech"}

        result = size_selected(survivors, config)

        assert len(result) == 2

        # A: clamped to 15% (pos cap), then scaled by 20/27
        a_expected = 15.0 * (20.0 / 27.0)
        # B: 12%, scaled by 20/27
        b_expected = 12.0 * (20.0 / 27.0)

        assert abs(result[0]["alloc"] - a_expected) < 0.01
        assert abs(result[1]["alloc"] - b_expected) < 0.01

        # Both should have POS_CAPPED and CLUSTER_CAPPED flags
        assert "POS_CAPPED" in result[0]["flags"]
        assert "CLUSTER_CAPPED" in result[0]["flags"]
        assert "POS_CAPPED" not in result[1]["flags"]
        assert "CLUSTER_CAPPED" in result[1]["flags"]

    def test_empty_survivors_list(self):
        """Empty input returns empty output."""
        config = AllocationConfig()
        result = size_selected([], config)
        assert result == []

    def test_all_rejected_rows(self):
        """All rejected rows pass through unchanged."""
        survivors = [
            {**self._make_survivor("A", kelly_frac=0.10), "status": "LOW_N"},
            {**self._make_survivor("B", kelly_frac=0.05), "status": "NEG_EV_NET"},
        ]
        config = AllocationConfig()

        result = size_selected(survivors, config)

        assert len(result) == 2
        assert result[0]["status"] == "LOW_N"
        assert result[1]["status"] == "NEG_EV_NET"
        # No alloc added to rejected rows
        assert "alloc" not in result[0]
        assert "alloc" not in result[1]

    def test_unmapped_tickers_form_singleton_clusters_not_shared_default(self):
        """Tickers not in cluster_map form their OWN singleton cluster (their ticker
        name), so unrelated unmapped tickers are never cluster-capped together."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.08),  # Not in cluster_map
            self._make_survivor("B", kelly_frac=0.09),  # Not in cluster_map
        ]
        # If A and B were incorrectly lumped into one shared "default" cluster,
        # their combined 17% would exceed a 10% cluster cap and get scaled down.
        # With correct singleton-cluster behavior, each is its own cluster and
        # neither individually exceeds the cap, so neither gets capped.
        config = AllocationConfig(
            max_pos_pct=15.0, max_cluster_pct=10.0, gross_cap_pct=100.0
        )
        config.cluster_map = {}  # Empty map

        result = size_selected(survivors, config)

        assert len(result) == 2
        assert result[0]["alloc"] == 8.0
        assert result[1]["alloc"] == 9.0
        assert "CLUSTER_CAPPED" not in result[0]["flags"]
        assert "CLUSTER_CAPPED" not in result[1]["flags"]
        assert "CLUSTER_CAPPED" not in result[1]["flags"]

    def test_dust_at_exactly_threshold(self):
        """Allocation exactly at dust_min_pct is NOT zeroed (>= check)."""
        survivors = [
            self._make_survivor("A", kelly_frac=0.01),  # Exactly 1.0%
        ]
        config = AllocationConfig(dust_min_pct=1.0)
        config.cluster_map = {}

        result = size_selected(survivors, config)

        assert len(result) == 1
        assert result[0]["alloc"] == 1.0
        assert result[0]["status"] == "SELECTED"  # Not DUST

    def test_input_not_mutated(self):
        """Original input list is not mutated."""
        survivor = self._make_survivor("A", kelly_frac=0.10, status=None)
        survivors = [survivor]

        # Make a copy to compare
        original_survivor = dict(survivor)

        config = AllocationConfig(max_pos_pct=15.0, max_cluster_pct=25.0, gross_cap_pct=100.0)
        config.cluster_map = {}

        result = size_selected(survivors, config)

        # Original should be unchanged
        assert survivor == original_survivor
        assert "alloc" not in survivor  # Original has no alloc
        assert "alloc" in result[0]  # Result has alloc


class TestAllocationResult:
    """Test AllocationResult dataclass."""

    def test_allocation_result_fields(self):
        """AllocationResult has all required fields."""
        result = AllocationResult(
            rows=[],
            selected_count=5,
            gross_exposure_pct=75.3,
            rejection_counts={"LOW_N": 10, "NEG_EV_NET": 5},
        )
        assert result.rows == []
        assert result.selected_count == 5
        assert result.gross_exposure_pct == 75.3
        assert result.rejection_counts == {"LOW_N": 10, "NEG_EV_NET": 5}

    def test_allocation_result_defaults(self):
        """AllocationResult has sensible defaults."""
        result = AllocationResult()
        assert result.rows == []
        assert result.selected_count == 0
        assert result.gross_exposure_pct == 0.0
        assert result.rejection_counts == {}


class TestAllocate:
    """Test allocate() orchestration function per RFC allocation_sheet.md §8."""

    def test_allocate_end_to_end_simple(self):
        """Simple end-to-end: one good candidate -> SELECTED."""
        candidates = [
            Candidate(
                strategy="s1",
                ticker="TEST",
                direction="long",
                entry=100.0,
                stop=95.0,
                target=110.0,
                ev_pct=5.0,
                base_win_rate=0.55,
                n=100,
                backtest_period="p",
                sharpe=1.0,
                advised_liquidity_pct=5.0,
            ),
        ]
        config = AllocationConfig()
        result = allocate(candidates, config)

        assert isinstance(result, AllocationResult)
        assert result.selected_count == 1
        assert len(result.rows) == 1
        assert result.rows[0]["status"] == "SELECTED"
        assert result.rows[0]["alloc"] > 0
        assert result.gross_exposure_pct == result.rows[0]["alloc"]

    def test_allocate_determinism(self):
        """Same input twice produces byte-identical rows per RFC §4.4."""
        candidates = [
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
            Candidate("s1", "B", "short", 100.0, 105.0, 90.0, 4.0, 0.52, 80, "p", 0.9, 5.0),
        ]
        config = AllocationConfig(top_k=2)

        result1 = allocate(candidates, config)
        result2 = allocate(candidates, config)

        # Both should have same rows in same order with same values
        assert len(result1.rows) == len(result2.rows)
        for r1, r2 in zip(result1.rows, result2.rows):
            assert r1["ticker"] == r2["ticker"]
            assert r1["status"] == r2["status"]
            assert r1["alloc"] == r2["alloc"]
            assert r1["derived"]["score"] == r2["derived"]["score"]

    def test_allocate_enabled_mask_defaults_to_all_enabled(self):
        """enabled_mask=None or empty dict defaults to all candidates enabled."""
        candidates = [
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
        ]
        config = AllocationConfig()

        # No enabled_mask (None)
        result1 = allocate(candidates, config)
        # Empty enabled_mask
        result2 = allocate(candidates, config, enabled_mask={})

        # Both should allow the candidate
        assert result1.rows[0]["status"] == "SELECTED"
        assert result2.rows[0]["status"] == "SELECTED"

    def test_allocate_enabled_mask_can_disable_ticker(self):
        """enabled_mask[ticker]=False disables that ticker."""
        candidates = [
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
        ]
        config = AllocationConfig()
        enabled_mask = {"A": False}

        result = allocate(candidates, config, enabled_mask)

        assert result.rows[0]["status"] == "DISABLED"
        assert result.selected_count == 0
        assert result.rejection_counts["DISABLED"] == 1

    def test_allocate_rejection_counts(self):
        """rejection_counts correctly aggregates all non-selected statuses."""
        candidates = [
            Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
            Candidate("s1", "B", "long", 100.0, 95.0, 110.0, 0.1, 0.55, 100, "p", 1.0, 5.0),
            Candidate("s1", "C", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 25, "p", 1.0, 5.0),
        ]
        config = AllocationConfig(top_k=1, min_n=50)

        result = allocate(candidates, config)

        # A: SELECTED, B: NEG_EV_NET, C: LOW_N
        assert result.selected_count == 1
        assert result.rejection_counts.get("NEG_EV_NET", 0) == 1
        assert result.rejection_counts.get("LOW_N", 0) == 1
        assert "SELECTED" not in result.rejection_counts  # SELECTED not in rejection_counts

    def test_allocate_gross_exposure_calculation(self):
        """gross_exposure_pct is sum of alloc for SELECTED rows."""
        candidates = [
            Candidate("s1", f"T{i}", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0)
            for i in range(3)
        ]
        config = AllocationConfig(top_k=3)
        config.cluster_map = {}

        result = allocate(candidates, config)

        # All three should be SELECTED with some allocation
        selected_allocs = [row["alloc"] for row in result.rows if row["status"] == "SELECTED"]
        expected_gross = sum(selected_allocs)
        assert abs(result.gross_exposure_pct - expected_gross) < 0.001

    def test_allocate_rfc_section_7_worked_example(self):
        """End-to-end test with RFC §7 worked example rows.

        Verifies:
        - NG=F close_direction wins DUP_ASSET collapse over NG=F open_gap
        - REMX carries DATA_MISMATCH flag
        - Score ranking: NG=F > REMX > V > CRM (only in selected, top_k limits)
        """
        # RFC §7 worked example (all with default config: n0=100, cost=0.15, kelly_mult=0.35)
        # NG=F close_direction: risk=0.6, reward=6.7, b=11.2, n=109, shrink=0.52, EV net=0.61%, Kelly=13.9%, Score=1.01
        # NG=F open_gap: risk=1.0, reward=5.4, b=5.4, n=79, shrink=0.44, EV net=0.56%, Kelly=12.1%, Score=0.56
        # REMX: risk=6.0, reward=7.6, b=1.27, n=161, shrink=0.62, EV net=2.34%, Kelly=2.5%, Score=0.39, DATA_MISMATCH
        # V: risk=1.6, reward=1.8, b=1.13, n=319, shrink=0.76, EV net=0.68%, Kelly=2.4%, Score=0.43
        # CRM: risk=1.5, reward=2.3, b=1.53, n=79, shrink=0.44, EV net=0.27%, Kelly=9.1%, Score=0.18

        candidates = [
            # NG=F close_direction: risk=0.6%, reward=6.7%
            # For long: stop = entry - entry*risk%, target = entry + entry*reward%
            Candidate(
                strategy="close_direction",
                ticker="NG=F",
                direction="long",
                entry=2.95,
                stop=2.95 - 2.95 * 0.006,  # risk 0.6% = 2.9323
                target=2.95 + 2.95 * 0.067,  # reward 6.7% = 3.14765
                ev_pct=1.45,  # Reported EV (will be shrunk to 0.755, then minus cost = 0.605)
                base_win_rate=0.538,
                n=109,
                backtest_period="test",
                sharpe=1.0,
                advised_liquidity_pct=0.0,
            ),
            # NG=F open_gap: risk=1.0%, reward=5.4% (duplicate ticker, lower score)
            Candidate(
                strategy="open_gap",
                ticker="NG=F",
                direction="long",
                entry=2.95,
                stop=2.95 - 2.95 * 0.01,  # risk 1.0% = 2.9205
                target=2.95 + 2.95 * 0.054,  # reward 5.4% = 3.109
                ev_pct=1.20,
                base_win_rate=0.505,
                n=79,
                backtest_period="test",
                sharpe=1.0,
                advised_liquidity_pct=0.0,
            ),
            # REMX
            Candidate(
                strategy="path_execution",
                ticker="REMX",
                direction="short",
                entry=79.73,
                stop=84.51,  # abs(84.51 - 79.73) / 79.73 * 100 ≈ 5.995% ≈ 6.0%
                target=73.71,  # abs(73.71 - 79.73) / 79.73 * 100 ≈ 7.533% ≈ 7.6%
                ev_pct=4.04,
                base_win_rate=0.47,
                n=161,
                backtest_period="test",
                sharpe=1.23,
                advised_liquidity_pct=11.0,
            ),
            # V: n=319, shrink=0.76, EV net=0.68%
            # Working backwards: ev_pct = (ev_net + cost) / shrink = (0.68 + 0.15) / 0.761 = 1.09%
            Candidate(
                strategy="strategy1",
                ticker="V",
                direction="long",
                entry=200.0,
                stop=196.8,  # 1.6%
                target=203.6,  # 1.8%
                ev_pct=1.09,
                base_win_rate=0.52,
                n=319,
                backtest_period="test",
                sharpe=1.0,
                advised_liquidity_pct=5.0,
            ),
            # CRM: n=79, shrink=0.44, EV net=0.27%
            # Working backwards: ev_pct = (ev_net + cost) / shrink = (0.27 + 0.15) / 0.44 = 0.955%
            Candidate(
                strategy="strategy2",
                ticker="CRM",
                direction="long",
                entry=250.0,
                stop=246.25,  # 1.5%
                target=255.75,  # 2.3%
                ev_pct=0.955,
                base_win_rate=0.48,
                n=79,
                backtest_period="test",
                sharpe=0.95,
                advised_liquidity_pct=5.0,
            ),
        ]

        config = AllocationConfig(top_k=12)  # All 5 can fit in top_k
        config.cluster_map = {}

        result = allocate(candidates, config)

        # Verify results
        assert len(result.rows) == 5

        # Find rows by ticker for easier verification
        rows_by_ticker = {row["ticker"]: row for row in result.rows}

        # NG=F close_direction should be SELECTED (wins collapse)
        ng_close = [r for r in result.rows if r["ticker"] == "NG=F" and r["strategy"] == "close_direction"][0]
        assert ng_close["status"] == "SELECTED", "NG=F close_direction should be SELECTED"

        # NG=F open_gap should be DUP_ASSET (loses to close_direction)
        ng_open = [r for r in result.rows if r["ticker"] == "NG=F" and r["strategy"] == "open_gap"][0]
        assert ng_open["status"] == "DUP_ASSET", "NG=F open_gap should be DUP_ASSET"

        # REMX should have DATA_MISMATCH flag
        remx = rows_by_ticker["REMX"]
        assert "DATA_MISMATCH" in remx["flags"], "REMX should have DATA_MISMATCH flag"
        assert remx["status"] == "SELECTED"

        # V should be SELECTED
        v = rows_by_ticker["V"]
        assert v["status"] == "SELECTED"

        # CRM should be SELECTED
        crm = rows_by_ticker["CRM"]
        assert crm["status"] == "SELECTED"

        # Verify score ordering (should win out NG=F > REMX > V > CRM)
        ng_close_score = ng_close["derived"]["score"]
        remx_score = remx["derived"]["score"]
        v_score = v["derived"]["score"]
        crm_score = crm["derived"]["score"]

        # Score ranking per RFC §7: NG=F close (1.01) > NG=F open (0.56) > V (0.43) > REMX (0.39) > CRM (0.18)
        # Note: REMX has higher raw EV but lower score due to high loss_pct (6.0% risk)
        assert ng_close_score > remx_score, f"NG=F score {ng_close_score} should exceed REMX {remx_score}"
        assert v_score > remx_score, f"V score {v_score} should exceed REMX {remx_score}"
        assert v_score > crm_score, f"V score {v_score} should exceed CRM {crm_score}"

        # Verify selected_count
        assert result.selected_count == 4  # NG=F close, REMX, V, CRM
        assert result.rejection_counts.get("DUP_ASSET", 0) == 1  # NG=F open_gap


class TestLoadClusterMap:
    """Test load_cluster_map() CSV file loader."""

    def test_load_cluster_map_basic(self):
        """Load simple CSV with two tickers."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("BTC,crypto\n")
            f.write("AAPL,tech\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {"BTC": "crypto", "AAPL": "tech"}
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_empty_file(self):
        """Load empty CSV returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_with_whitespace(self):
        """Whitespace around ticker/cluster is stripped."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("  BTC  ,  crypto  \n")
            f.write("AAPL,tech\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {"BTC": "crypto", "AAPL": "tech"}
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_skips_empty_ticker_lines(self):
        """Empty ticker lines (blank lines) are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("BTC,crypto\n")
            f.write("  ,energy\n")  # Empty ticker
            f.write("AAPL,tech\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {"BTC": "crypto", "AAPL": "tech"}
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_too_few_columns_raises(self):
        """Line with fewer than 2 columns raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("BTC,crypto\n")
            f.write("AAPL\n")  # Missing cluster column
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="fewer than 2 columns"):
                load_cluster_map(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_file_not_found_raises(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_cluster_map("/nonexistent/path/to/file.csv")

    def test_load_cluster_map_multiple_rows(self):
        """Load CSV with multiple tickers and clusters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("BTC,crypto\n")
            f.write("ETH,crypto\n")
            f.write("AAPL,tech\n")
            f.write("MSFT,tech\n")
            f.write("XOM,energy\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {
                "BTC": "crypto",
                "ETH": "crypto",
                "AAPL": "tech",
                "MSFT": "tech",
                "XOM": "energy",
            }
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_duplicates_last_wins(self):
        """Duplicate ticker entries, last one wins."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("BTC,crypto\n")
            f.write("BTC,asset_class_old\n")  # Duplicate, should be overwritten
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {"BTC": "asset_class_old"}  # Last entry wins
        finally:
            os.unlink(temp_path)

    def test_load_cluster_map_extra_columns_ignored(self):
        """Extra columns beyond the first two are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("BTC,crypto,extra1,extra2\n")
            f.write("AAPL,tech,metadata\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_cluster_map(temp_path)
            assert result == {"BTC": "crypto", "AAPL": "tech"}
        finally:
            os.unlink(temp_path)

    def test_allocate_with_loaded_cluster_map(self):
        """Integration test: load_cluster_map used in allocate() sizing."""
        # Create cluster map file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("A,tech\n")
            f.write("B,tech\n")
            f.write("C,energy\n")
            f.flush()
            temp_path = f.name

        try:
            cluster_map = load_cluster_map(temp_path)

            candidates = [
                Candidate("s1", "A", "long", 100.0, 95.0, 110.0, 5.0, 0.55, 100, "p", 1.0, 5.0),
                Candidate("s1", "B", "long", 100.0, 95.0, 110.0, 4.5, 0.55, 100, "p", 1.0, 5.0),
                Candidate("s1", "C", "long", 100.0, 95.0, 110.0, 4.0, 0.55, 100, "p", 1.0, 5.0),
            ]

            config = AllocationConfig(
                top_k=3,
                max_pos_pct=15.0,
                max_cluster_pct=20.0,  # Tech cluster: A+B will exceed this
                gross_cap_pct=100.0,
            )
            config.cluster_map = cluster_map

            result = allocate(candidates, config)

            # All three should be in top_k
            assert result.selected_count == 3

            # Tech cluster should be capped; A and B should have CLUSTER_CAPPED flags
            a = [r for r in result.rows if r["ticker"] == "A"][0]
            b = [r for r in result.rows if r["ticker"] == "B"][0]
            c = [r for r in result.rows if r["ticker"] == "C"][0]

            assert "CLUSTER_CAPPED" in a["flags"]
            assert "CLUSTER_CAPPED" in b["flags"]
            assert "CLUSTER_CAPPED" not in c["flags"]  # Energy cluster not capped

        finally:
            os.unlink(temp_path)
