"""test_asset_class_wiring.py — Unit tests for asset_class threading through signal selection.

asset_class is derived from the candidate's own TICKER (asset_class_of_symbol,
the 3-way per-symbol classifier strategy_class_stats is keyed on), NOT from
viability_report's 5-way group-majority label that arrives in stats_row. See
tests/unit/test_candidate_asset_class.py.

Tests that asset_class flows through:
  - STATS_COLUMNS (kairos_signals.py)
  - stats_rows dict (kairos_signals.py:920-946)
  - Candidate dataclass (allocation.py:20-43)
  - DSL column registry (signal_selection.py:74-99)

Ensures selection rules can filter on 'Asset Class' and that missing asset_class
doesn't crash rule evaluation.
"""

import pytest

from allocation import Candidate, fetch_signals, AllocationConfig, compute_derived
from signal_selection import (
    parse_signal_selection,
    rule_matches,
)


class TestAssetClassInCandidate:
    """asset_class flows from stats_rows through Candidate objects."""

    def test_asset_class_survives_to_candidate(self):
        """asset_class field is populated in Candidate from stats_rows."""
        stats_rows = [
            {
                "strategy": "momentum",
                "symbol": "BTC-USD",
                "direction": "LONG",
                "entry": 50000.0,
                "stop": 48000.0,
                "target": 55000.0,
                "expected_value": 2000.0,
                "base_sharpe": 1.5,
                "base_win_rate": 0.55,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.10,
                "asset_class": "crypto",
            }
        ]
        advice_rows = [
            {
                "expected_value": 2000.0,
                "entry": 50000.0,
                "base_win_rate": 0.55,
                "base_signals": 100,
                "oracle_signals": None,
                "signal": "Buy BTC",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.asset_class == "crypto"

    def test_asset_class_derived_from_ticker_when_absent_from_stats_row(self):
        """stats_row's asset_class is not consulted at all -- the ticker decides."""
        stats_rows = [
            {
                "strategy": "momentum",
                "symbol": "MSFT",
                "direction": "LONG",
                "entry": 300.0,
                "stop": 290.0,
                "target": 310.0,
                "expected_value": 10.0,
                "base_sharpe": 1.2,
                "base_win_rate": 0.60,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.05,
                # asset_class not provided
            }
        ]
        advice_rows = [
            {
                "expected_value": 10.0,
                "entry": 300.0,
                "base_win_rate": 0.60,
                "base_signals": 80,
                "oracle_signals": None,
                "signal": "Buy MSFT",
            }
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.asset_class == "equity"  # from the MSFT ticker

    def test_multiple_asset_classes_in_candidates(self):
        """Multiple candidates with different asset_classes are preserved."""
        stats_rows = [
            {
                "strategy": "momentum",
                "symbol": "BTC-USD",
                "direction": "LONG",
                "entry": 50000.0,
                "stop": 48000.0,
                "target": 55000.0,
                "expected_value": 2000.0,
                "base_sharpe": 1.5,
                "base_win_rate": 0.55,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.10,
                "asset_class": "crypto",
            },
            {
                "strategy": "momentum",
                "symbol": "MSFT",
                "direction": "LONG",
                "entry": 300.0,
                "stop": 290.0,
                "target": 310.0,
                "expected_value": 10.0,
                "base_sharpe": 1.2,
                "base_win_rate": 0.60,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.05,
                "asset_class": "equity",
            },
            {
                "strategy": "momentum",
                "symbol": "EURUSD=X",
                "direction": "SHORT",
                "entry": 1.0800,
                "stop": 1.0900,
                "target": 1.0700,
                "expected_value": 0.0050,
                "base_sharpe": 0.9,
                "base_win_rate": 0.52,
                "backtest_period": "2023-01-01 to 2023-12-31",
                "size": 0.03,
                "asset_class": "fx_commodity",
            },
        ]
        advice_rows = [
            {
                "expected_value": 2000.0,
                "entry": 50000.0,
                "base_win_rate": 0.55,
                "base_signals": 100,
                "oracle_signals": None,
                "signal": "Buy BTC",
            },
            {
                "expected_value": 10.0,
                "entry": 300.0,
                "base_win_rate": 0.60,
                "base_signals": 80,
                "oracle_signals": None,
                "signal": "Buy MSFT",
            },
            {
                "expected_value": 0.0050,
                "entry": 1.0800,
                "base_win_rate": 0.52,
                "base_signals": 60,
                "oracle_signals": None,
                "signal": "Sell EURUSD=X",
            },
        ]
        candidates = fetch_signals(stats_rows, advice_rows)

        assert len(candidates) == 3
        assert candidates[0].asset_class == "crypto"
        assert candidates[1].asset_class == "equity"
        assert candidates[2].asset_class == "fx_commodity"


class TestAssetClassInDSL:
    """DSL rules can filter on 'Asset Class' column."""

    def test_asset_class_column_in_registry(self):
        """'Asset Class' column is registered in the DSL."""
        rule = parse_signal_selection("'Asset Class' == 'crypto'")
        assert len(rule.conditions) == 1
        cond = rule.conditions[0]
        assert cond.column == "Asset Class"
        assert cond.op == "=="
        assert cond.value == "crypto"

    def test_asset_class_filter_selects_crypto_only(self):
        """Rule filtering on 'Asset Class' == 'crypto' selects only crypto candidates."""
        crypto_candidate = Candidate(
            strategy="momentum",
            ticker="BTC-USD",
            direction="long",
            entry=50000.0,
            stop=48000.0,
            target=55000.0,
            ev_pct=4.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.5,
            advised_liquidity_pct=10.0,
            asset_class="crypto",
        )
        equity_candidate = Candidate(
            strategy="momentum",
            ticker="MSFT",
            direction="long",
            entry=300.0,
            stop=290.0,
            target=310.0,
            ev_pct=3.33,
            base_win_rate=0.60,
            n=80,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.2,
            advised_liquidity_pct=5.0,
            asset_class="equity",
        )

        rule = parse_signal_selection("'Asset Class' == 'crypto'")
        config = AllocationConfig()
        derived_crypto = compute_derived(crypto_candidate, config)
        derived_equity = compute_derived(equity_candidate, config)

        matched_crypto, _ = rule_matches(rule, crypto_candidate, derived_crypto, {})
        matched_equity, _ = rule_matches(rule, equity_candidate, derived_equity, {})

        assert matched_crypto
        assert not matched_equity

    def test_asset_class_filter_not_equal(self):
        """Rule filtering on 'Asset Class' != 'crypto' excludes crypto candidates."""
        crypto_candidate = Candidate(
            strategy="momentum",
            ticker="BTC-USD",
            direction="long",
            entry=50000.0,
            stop=48000.0,
            target=55000.0,
            ev_pct=4.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.5,
            advised_liquidity_pct=10.0,
            asset_class="crypto",
        )
        equity_candidate = Candidate(
            strategy="momentum",
            ticker="MSFT",
            direction="long",
            entry=300.0,
            stop=290.0,
            target=310.0,
            ev_pct=3.33,
            base_win_rate=0.60,
            n=80,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.2,
            advised_liquidity_pct=5.0,
            asset_class="equity",
        )

        rule = parse_signal_selection("'Asset Class' != 'crypto'")
        config = AllocationConfig()
        derived_crypto = compute_derived(crypto_candidate, config)
        derived_equity = compute_derived(equity_candidate, config)

        matched_crypto, _ = rule_matches(rule, crypto_candidate, derived_crypto, {})
        matched_equity, _ = rule_matches(rule, equity_candidate, derived_equity, {})

        assert not matched_crypto
        assert matched_equity

    def test_asset_class_combined_with_other_conditions(self):
        """Rule with asset_class and numeric conditions filters correctly."""
        crypto_high_sharpe = Candidate(
            strategy="momentum",
            ticker="BTC-USD",
            direction="long",
            entry=50000.0,
            stop=48000.0,
            target=55000.0,
            ev_pct=4.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=2.0,
            advised_liquidity_pct=10.0,
            asset_class="crypto",
        )
        crypto_low_sharpe = Candidate(
            strategy="momentum",
            ticker="ETH-USD",
            direction="long",
            entry=3000.0,
            stop=2900.0,
            target=3100.0,
            ev_pct=3.33,
            base_win_rate=0.50,
            n=80,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=0.8,
            advised_liquidity_pct=5.0,
            asset_class="crypto",
        )

        # Rule: crypto AND sharpe > 1.0
        rule = parse_signal_selection("'Asset Class' == 'crypto', 'Sharpe' > 1.0")
        config = AllocationConfig()
        derived_high = compute_derived(crypto_high_sharpe, config)
        derived_low = compute_derived(crypto_low_sharpe, config)

        matched_high, _ = rule_matches(rule, crypto_high_sharpe, derived_high, {})
        matched_low, _ = rule_matches(rule, crypto_low_sharpe, derived_low, {})

        assert matched_high
        assert not matched_low


class TestAssetClassRuleBackwardCompatibility:
    """Rules not mentioning asset_class are unaffected."""

    def test_rule_without_asset_class_unaffected(self):
        """Existing rules that don't mention 'Asset Class' work unchanged."""
        candidate = Candidate(
            strategy="momentum",
            ticker="BTC-USD",
            direction="long",
            entry=50000.0,
            stop=48000.0,
            target=55000.0,
            ev_pct=4.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.5,
            advised_liquidity_pct=10.0,
            asset_class="crypto",
        )

        # Rule with no asset_class condition
        rule = parse_signal_selection("'Sharpe' > 1.0, 'n' > 50")
        config = AllocationConfig()
        derived = compute_derived(candidate, config)

        matched, _ = rule_matches(rule, candidate, derived, {})
        assert matched

    def test_rule_on_other_text_columns_unaffected(self):
        """Existing rules on other text columns (Ticker, Strategy, Dir) still work."""
        candidate = Candidate(
            strategy="momentum",
            ticker="BTC-USD",
            direction="long",
            entry=50000.0,
            stop=48000.0,
            target=55000.0,
            ev_pct=4.0,
            base_win_rate=0.55,
            n=100,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.5,
            advised_liquidity_pct=10.0,
            asset_class="crypto",
        )

        # Rule on different text columns
        rule = parse_signal_selection("'Strategy' == 'momentum', 'Dir' == 'long'")
        config = AllocationConfig()
        derived = compute_derived(candidate, config)

        matched, _ = rule_matches(rule, candidate, derived, {})
        assert matched


class TestAssetClassMissingHandling:
    """Missing or empty asset_class doesn't crash rule evaluation."""

    def test_empty_asset_class_does_not_crash_rule(self):
        """Candidate with empty asset_class (from missing data) doesn't crash."""
        candidate = Candidate(
            strategy="momentum",
            ticker="MSFT",
            direction="long",
            entry=300.0,
            stop=290.0,
            target=310.0,
            ev_pct=3.33,
            base_win_rate=0.60,
            n=80,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.2,
            advised_liquidity_pct=5.0,
            asset_class="",  # Empty because missing from stats_row
        )

        rule = parse_signal_selection("'Asset Class' == 'crypto'")
        config = AllocationConfig()
        derived = compute_derived(candidate, config)

        # Should not crash, should simply not match
        matched, _ = rule_matches(rule, candidate, derived, {})
        assert not matched

    def test_none_asset_class_does_not_crash_rule(self):
        """Candidate with None asset_class doesn't crash."""
        candidate = Candidate(
            strategy="momentum",
            ticker="MSFT",
            direction="long",
            entry=300.0,
            stop=290.0,
            target=310.0,
            ev_pct=3.33,
            base_win_rate=0.60,
            n=80,
            backtest_period="2023-01-01 to 2023-12-31",
            sharpe=1.2,
            advised_liquidity_pct=5.0,
            asset_class=None,
        )

        rule = parse_signal_selection("'Asset Class' == 'crypto'")
        config = AllocationConfig()
        derived = compute_derived(candidate, config)

        # Should not crash, should simply not match
        matched, _ = rule_matches(rule, candidate, derived, {})
        assert not matched

    def test_rule_without_asset_class_ignores_missing_values(self):
        """Rules not mentioning asset_class work regardless of missing asset_class."""
        candidates = [
            Candidate(
                strategy="momentum",
                ticker="BTC-USD",
                direction="long",
                entry=50000.0,
                stop=48000.0,
                target=55000.0,
                ev_pct=4.0,
                base_win_rate=0.55,
                n=100,
                backtest_period="2023-01-01 to 2023-12-31",
                sharpe=1.5,
                advised_liquidity_pct=10.0,
                asset_class="crypto",
            ),
            Candidate(
                strategy="momentum",
                ticker="MSFT",
                direction="long",
                entry=300.0,
                stop=290.0,
                target=310.0,
                ev_pct=3.33,
                base_win_rate=0.60,
                n=80,
                backtest_period="2023-01-01 to 2023-12-31",
                sharpe=1.2,
                advised_liquidity_pct=5.0,
                asset_class="",  # Empty
            ),
            Candidate(
                strategy="momentum",
                ticker="EUR/USD",
                direction="short",
                entry=1.0800,
                stop=1.0900,
                target=1.0700,
                ev_pct=0.46,
                base_win_rate=0.52,
                n=60,
                backtest_period="2023-01-01 to 2023-12-31",
                sharpe=0.9,
                advised_liquidity_pct=3.0,
                asset_class=None,  # None
            ),
        ]

        rule = parse_signal_selection("'Sharpe' > 0.5, 'n' > 50")
        config = AllocationConfig()

        # All three should match the rule even though asset_class varies
        for candidate in candidates:
            derived = compute_derived(candidate, config)
            matched, _ = rule_matches(rule, candidate, derived, {})
            assert matched
