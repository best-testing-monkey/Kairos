"""Tests for asset_class_for() and resolve_disabled_strategies() in
kairos_strategies.py: pure logic (plus a tmp-path SQLite DB for the
DB-backed profile lookup), no network/model access.
"""
import sys, os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest

import kairos_pipeline
from kairos_strategies import (
    asset_class_for,
    resolve_disabled_strategies,
    _DISABLED_BY_CLASS,
)


class TestAssetClassFor:
    def test_crypto(self):
        assert asset_class_for(["BTC-USD", "ETH-USD", "SOL-USD"]) == "crypto"

    def test_fx(self):
        assert asset_class_for(["EURUSD=X", "GBPUSD=X", "USDCHF=X"]) == "fx"

    def test_commodity_futures(self):
        assert asset_class_for(["CL=F", "USO"]) == "commodity"

    def test_commodity_etfs(self):
        assert asset_class_for(["GLD", "SLV", "GDX"]) == "commodity"

    def test_mixed_commodity_group(self):
        # COPX, GC=F, GDX, GLD - all commodity-classified symbols
        assert asset_class_for(["COPX", "GC=F", "GDX", "GLD"]) == "commodity"

    def test_equity(self):
        assert asset_class_for(["AAPL", "MSFT", "SPY"]) == "equity"

    def test_mixed_no_majority(self):
        # 2 crypto, 2 equity: no strict majority -> mixed
        assert asset_class_for(["BTC-USD", "ETH-USD", "AAPL", "MSFT"]) == "mixed"

    def test_empty_list_is_mixed(self):
        assert asset_class_for([]) == "mixed"

    def test_majority_wins(self):
        # 3 equity, 1 crypto -> majority equity
        assert asset_class_for(["AAPL", "MSFT", "GOOG", "BTC-USD"]) == "equity"


@pytest.fixture
def temp_db(tmp_path):
    """A tmp-path SQLite DB with the pipeline schema, for seeding
    oracle_results/disabled_strategies rows directly via raw SQL."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(kairos_pipeline.SCHEMA)
    conn.commit()
    conn.close()
    return db_path


def _seed_oracle_row(db_path, interval, assets_key, strategy_name="some_strategy"):
    """Insert a minimal oracle_results row so the (interval, assets_key)
    profile counts as 'tested' for resolve_disabled_strategies."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO oracle_results
           (run_id, stage, strategy_name, sharpe, signal_count, win_rate,
            avg_pnl_per_trade, assets, interval, backtest_period)
           VALUES (1, 'oracle', ?, 1.0, 10, 0.5, 0.01, ?, ?, '6m')""",
        (strategy_name, assets_key, interval),
    )
    conn.commit()
    conn.close()


def _seed_disabled_rows(db_path, interval, assets_key, strategy_names):
    conn = sqlite3.connect(db_path)
    for name in strategy_names:
        conn.execute(
            """INSERT INTO disabled_strategies
               (interval, assets, strategy_name, avg_pnl_per_trade, sharpe,
                signal_count, source_run_id, updated_at)
               VALUES (?,?,?,-0.01,-1.0,10,1,'2026-01-01T00:00:00')""",
            (interval, assets_key, name),
        )
    conn.commit()
    conn.close()


class TestResolveDisabledStrategies:
    def test_exact_profile_wins_over_class(self, temp_db):
        # BTC-USD,ETH-USD,SOL-USD would fall into the (1d, crypto) class
        # bucket, but a seeded, differing DB profile must win instead.
        assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
        assets_key = ",".join(sorted(assets))
        seeded = {"my_custom_disabled_strategy"}
        _seed_oracle_row(temp_db, "1d", assets_key)
        _seed_disabled_rows(temp_db, "1d", assets_key, seeded)

        result = resolve_disabled_strategies("1d", assets, db_path=temp_db)

        assert result == seeded
        assert result != _DISABLED_BY_CLASS.get(("1d", "crypto"))

    def test_exact_profile_independent_of_asset_order(self, temp_db):
        assets_key = "BTC-USD,ETH-USD,SOL-USD"
        seeded = {"my_custom_disabled_strategy"}
        _seed_oracle_row(temp_db, "1d", assets_key)
        _seed_disabled_rows(temp_db, "1d", assets_key, seeded)

        result = resolve_disabled_strategies("1d", ["SOL-USD", "BTC-USD", "ETH-USD"], db_path=temp_db)

        assert result == seeded

    def test_tested_but_clean_returns_empty_not_class_fallback(self, temp_db):
        # Profile is oracle-tested but has no disabled_strategies rows -
        # must return set(), NOT fall back to the (1d, crypto) class bucket
        # (which is non-empty and contains "volume_fade").
        assets = ["ADA-USD", "DOGE-USD", "DOT-USD", "LINK-USD"]
        assets_key = ",".join(sorted(assets))
        _seed_oracle_row(temp_db, "1d", assets_key)
        # No disabled_strategies rows seeded for this profile.

        result = resolve_disabled_strategies("1d", assets, db_path=temp_db)

        assert result == set()
        class_fallback = _DISABLED_BY_CLASS[("1d", "crypto")]
        assert "volume_fade" in class_fallback
        assert "volume_fade" not in result
        assert result != class_fallback

    def test_class_fallback_when_no_exact_profile(self, temp_db):
        # No oracle_results row at all for this profile -> class fallback.
        assets = ["ADA-USD", "DOGE-USD", "DOT-USD", "LINK-USD"]
        result = resolve_disabled_strategies("1d", assets, db_path=temp_db)
        assert result == _DISABLED_BY_CLASS[("1d", "crypto")]
        assert "volume_fade" in result

    def test_class_fallback_for_equity(self, temp_db):
        assets = ["AAPL", "MSFT", "GOOG"]
        result = resolve_disabled_strategies("1d", assets, db_path=temp_db)
        assert result == _DISABLED_BY_CLASS[("1d", "equity")]

    def test_missing_db_falls_back_to_class_no_crash(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist" / "pipeline_results.db")
        assets = ["AAPL", "MSFT", "GOOG"]
        result = resolve_disabled_strategies("1d", assets, db_path=nonexistent)
        assert result == _DISABLED_BY_CLASS[("1d", "equity")]

    def test_empty_fallback_for_unknown_interval(self, temp_db):
        assets = ["BTC-USD", "ETH-USD"]
        result = resolve_disabled_strategies("15m", assets, db_path=temp_db)
        assert result == set()

    def test_empty_fallback_for_mixed_class_without_bucket(self, temp_db):
        assets = ["BTC-USD", "AAPL"]
        assert asset_class_for(assets) == "mixed"
        result = resolve_disabled_strategies("1d", assets, db_path=temp_db)
        assert result == set()
