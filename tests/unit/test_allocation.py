"""test_allocation.py — Unit tests for allocation.Candidate and fetch_signals().

Tests the schema and fetch adapter without GPU/network, using hand-constructed
fixtures that mirror the exact dict shapes from kairos_signals.py run().
"""

import math
import pytest
from allocation import Candidate, fetch_signals, validate_candidate


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
