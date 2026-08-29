"""Tests for the new (item 1+2) per-(model, class) disabled-strategy gate in
resolve_disabled_strategies() (strategy/kairos_strategies.py), sourced from
strategy_class_stats. See docs/tickets/per-class-stats-wiring.md.

Does NOT touch/duplicate tests/unit/test_disabled_strategy_resolution.py -
that file pins the pre-existing 3-step (now 4-step) contract and must keep
passing unmodified.
"""
import sys, os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest

import kairos_pipeline
from kairos_strategies import resolve_disabled_strategies, _DISABLED_BY_CLASS


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(kairos_pipeline.SCHEMA)
    conn.commit()
    conn.close()
    return db_path


def _seed_oracle_row(db_path, interval, assets_key, strategy_name="some_strategy"):
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


def _seed_class_stats(db_path, rows):
    """rows: list of dicts with keys strategy_name, asset_class, stage,
    model_path, avg_pnl_per_trade, signal_count, interval; run_id/sharpe/
    win_rate/assets/backtest_period/version filled with harmless defaults."""
    conn = sqlite3.connect(db_path)
    for i, r in enumerate(rows):
        conn.execute(
            """INSERT INTO strategy_class_stats
               (run_id, stage, model_path, strategy_name, asset_class, sharpe,
                signal_count, win_rate, avg_pnl_per_trade, assets, interval,
                backtest_period, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (i, r["stage"], r.get("model_path"), r["strategy_name"], r["asset_class"],
             r.get("sharpe", -1.0), r["signal_count"], r.get("win_rate", 0.4),
             r["avg_pnl_per_trade"], r.get("assets", "X"), r["interval"],
             r.get("backtest_period", "6m"), r.get("version", "abc123")),
        )
    conn.commit()
    conn.close()


CRYPTO_ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]


class TestClassStatsGate:
    def test_oracle_tested_profile_wins_over_class_stats_gate(self, temp_db):
        # Even though strategy_class_stats would disable "some_strategy" for
        # this exact class/model cell, an oracle-tested profile (step 1)
        # must win outright - the new gate is never consulted.
        assets_key = ",".join(sorted(CRYPTO_ASSETS))
        _seed_oracle_row(temp_db, "1d", assets_key)
        _seed_disabled_rows(temp_db, "1d", assets_key, {"only_this_one"})
        _seed_class_stats(temp_db, [
            {"strategy_name": "some_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -5.0, "signal_count": 100,
             "interval": "1d"},
        ])

        result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)

        assert result == {"only_this_one"}

    def test_tested_but_clean_does_not_fall_through_to_class_stats_gate(self, temp_db):
        assets_key = ",".join(sorted(CRYPTO_ASSETS))
        _seed_oracle_row(temp_db, "1d", assets_key)
        # No disabled_strategies rows -> tested-but-clean.
        _seed_class_stats(temp_db, [
            {"strategy_name": "some_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -5.0, "signal_count": 100,
             "interval": "1d"},
        ])

        result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)

        assert result == set()

    def test_class_stats_gate_disables_strategy_for_pure_class_group(self, temp_db):
        # No oracle row at all -> step 1 falls through; a pure-crypto group
        # with >=30 weighted-negative signals should be disabled by the new
        # gate rather than the hardcoded _DISABLED_BY_CLASS dict.
        _seed_class_stats(temp_db, [
            {"strategy_name": "bad_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -2.0, "signal_count": 40,
             "interval": "1d"},
            {"strategy_name": "good_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": 3.0, "signal_count": 40,
             "interval": "1d"},
        ])

        result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)

        assert result == {"bad_strategy"}
        # Confirms the hardcoded dict was NOT used (it disables a very
        # different, larger set for ("1d", "crypto")).
        assert result != _DISABLED_BY_CLASS[("1d", "crypto")]

    def test_thin_cell_falls_through_to_hardcoded_dict(self, temp_db):
        # No row at all for the (interval, class, stage, model) cell being
        # queried (everything seeded is under interval="4h") -> the query
        # returns zero rows, so it must fall through to the hardcoded dict.
        _seed_class_stats(temp_db, [
            {"strategy_name": "bad_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -2.0, "signal_count": 10,
             "interval": "4h"},
        ])

        result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)

        assert result == _DISABLED_BY_CLASS[("1d", "crypto")]

    def test_swept_but_thin_cell_falls_through_rather_than_disabling_nothing(self, temp_db):
        # A matching cell exists but NO strategy in it clears the 30-signal
        # threshold. The cell is not authoritative: returning its (empty)
        # disabled set would silently drop the hardcoded safety net for this
        # class, which is exactly the failure this gate exists to avoid.
        _seed_class_stats(temp_db, [
            {"strategy_name": "bad_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -2.0, "signal_count": 10,
             "interval": "1d"},
        ])

        result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)

        assert result == _DISABLED_BY_CLASS[("1d", "crypto")]
        assert result != set()

    def test_one_thick_strategy_makes_the_cell_authoritative(self, temp_db):
        # Mixed thickness: one strategy clears the threshold, so the cell IS
        # judged. The thin one is simply not disabled (insufficient evidence),
        # and the hardcoded fallback does not apply.
        _seed_class_stats(temp_db, [
            {"strategy_name": "bad_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -2.0, "signal_count": 500,
             "interval": "1d"},
            {"strategy_name": "thin_loser", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -9.0, "signal_count": 3,
             "interval": "1d"},
        ])

        result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)

        assert result == {"bad_strategy"}

    def test_mixed_class_group_skips_db_gate_falls_to_hardcoded_dict(self, temp_db):
        # BTC-USD -> crypto, EURUSD=X/GBPUSD=X -> fx_commodity (3-way
        # symbol classifier): spans 2 classes, so step 2 must be skipped
        # entirely. asset_class_for() (5-way, per-group majority) rates
        # this group "fx" (2 of 3 symbols), which is NOT mixed at the
        # group level - so step 3's hardcoded dict is exercised cleanly.
        assets = ["BTC-USD", "EURUSD=X", "GBPUSD=X"]
        _seed_class_stats(temp_db, [
            # If step 2 were wrongly applied to the crypto symbol alone,
            # this row would disable "sneaky_strategy" instead of falling
            # through to the hardcoded ("1d", "fx") set.
            {"strategy_name": "sneaky_strategy", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -9.0, "signal_count": 100,
             "interval": "1d"},
        ])

        result = resolve_disabled_strategies("1d", assets, db_path=temp_db)

        assert result == _DISABLED_BY_CLASS[("1d", "fx")]
        assert "sneaky_strategy" not in result

    def test_model_path_distinguishes_base_from_finetuned(self, temp_db):
        _seed_class_stats(temp_db, [
            {"strategy_name": "model_sensitive", "asset_class": "crypto", "stage": "base",
             "model_path": None, "avg_pnl_per_trade": -2.0, "signal_count": 50,
             "interval": "1d"},
            {"strategy_name": "model_sensitive", "asset_class": "crypto", "stage": "finetuned",
             "model_path": "models/ft-abc", "avg_pnl_per_trade": 4.0, "signal_count": 50,
             "interval": "1d"},
        ])

        base_result = resolve_disabled_strategies("1d", CRYPTO_ASSETS, db_path=temp_db)
        ft_result = resolve_disabled_strategies(
            "1d", CRYPTO_ASSETS, db_path=temp_db, model_path="models/ft-abc",
        )

        assert base_result == {"model_sensitive"}
        assert ft_result == set()

    def test_sqlite_error_falls_back_to_class_dict(self, tmp_path):
        # A file that exists but isn't a valid SQLite DB: sqlite3.connect()
        # succeeds lazily, but the first execute() raises sqlite3.Error.
        bad_db = tmp_path / "corrupt.db"
        bad_db.write_text("not a real sqlite database")

        result = resolve_disabled_strategies("1d", ["AAPL", "MSFT", "GOOG"], db_path=str(bad_db))

        assert result == _DISABLED_BY_CLASS[("1d", "equity")]
