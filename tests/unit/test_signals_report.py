import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

from kairos_backtest import Direction, Signal
from kairos_signals import (
    signal_to_advice,
    render_report,
    load_work_items,
    group_items,
    build_strategy_index,
    run,
    format_table,
    _format_numeric_cell,
    _format_ev_pct,
)


# ============================================================================
# signal_to_advice
# ============================================================================

class TestSignalToAdvice:
    def test_long_with_sl_tp(self):
        sig = Signal(
            direction=Direction.LONG, size=0.12, entry=60800.0,
            stop=58900.0, target=63400.0, strategy_name="dfa_persistence",
            confidence=0.8, expected_value=0.02,
        )
        result = signal_to_advice("dfa_persistence", "BTC-USD", sig)
        assert result == (
            "Strategy dfa_persistence advised **Long** position on BTC-USD "
            "for 12% liquidity with SL at 58,900.00 (-3.1%) and TP at "
            "63,400.00 (+4.3%). Exit by TP/SL."
        )

    def test_short_with_sl_tp(self):
        sig = Signal(
            direction=Direction.SHORT, size=0.20, entry=100.0,
            stop=105.0, target=90.0, strategy_name="range_trading",
            confidence=0.5, expected_value=0.01,
        )
        result = signal_to_advice("range_trading", "ETH-USD", sig)
        assert result == (
            "Strategy range_trading advised **Short** position on ETH-USD "
            "for 20% liquidity with SL at 105.00 (+5.0%) and TP at "
            "90.00 (-10.0%). Exit by TP/SL."
        )

    def test_flat(self):
        sig = Signal(
            direction=Direction.FLAT, size=0.0, entry=100.0,
            stop=0.0, target=0.0, strategy_name="rqa_determinism",
            confidence=0.0, expected_value=0.0,
        )
        result = signal_to_advice("rqa_determinism", "AAPL", sig)
        assert result == "Strategy rqa_determinism advised **Exit/Flat** on AAPL."

    def test_missing_stop_target(self):
        sig = Signal(
            direction=Direction.LONG, size=0.15, entry=50.0,
            stop=None, target=None, strategy_name="momentum",
            confidence=0.6, expected_value=0.03,
        )
        result = signal_to_advice("momentum", "SOL-USD", sig)
        assert result == (
            "Strategy momentum advised **Long** position on SOL-USD "
            "for 15% liquidity. Exit on momentum exit signal."
        )

    def test_nan_stop_target(self):
        import numpy as np
        sig = Signal(
            direction=Direction.SHORT, size=0.10, entry=50.0,
            stop=float("nan"), target=np.nan, strategy_name="skew",
            confidence=0.4, expected_value=0.01,
        )
        result = signal_to_advice("skew", "XRP-USD", sig)
        assert result == (
            "Strategy skew advised **Short** position on XRP-USD "
            "for 10% liquidity. Exit on skew exit signal."
        )


# ============================================================================
# _format_numeric_cell
# ============================================================================

class TestFormatNumericCell:
    def test_format_numeric_cell_with_2_decimals(self):
        """_format_numeric_cell should format with 2 decimals by default."""
        assert _format_numeric_cell(0.7654) == "0.77"
        assert _format_numeric_cell(0.123) == "0.12"
        assert _format_numeric_cell(100.456) == "100.46"

    def test_format_numeric_cell_missing_values(self):
        """_format_numeric_cell should return empty string for None or NaN."""
        import numpy as np
        assert _format_numeric_cell(None) == ""
        assert _format_numeric_cell(float("nan")) == ""
        assert _format_numeric_cell(np.nan) == ""

    def test_format_numeric_cell_custom_decimals(self):
        """_format_numeric_cell should support custom decimal places."""
        assert _format_numeric_cell(0.123456, decimals=4) == "0.1235"
        assert _format_numeric_cell(0.123456, decimals=1) == "0.1"


class TestFormatEvPct:
    def test_format_ev_pct_basic(self):
        """_format_ev_pct should compute EV as percentage of entry price."""
        # entry 100, EV 0.9 → 0.9% → +0.90%
        assert _format_ev_pct(0.9, 100.0) == "+0.90%"
        # entry 100, EV -2.5 → -2.5% → -2.50%
        assert _format_ev_pct(-2.5, 100.0) == "-2.50%"
        # entry 50, EV 5.0 → 10% → +10.00%
        assert _format_ev_pct(5.0, 50.0) == "+10.00%"

    def test_format_ev_pct_zero_entry(self):
        """_format_ev_pct should return blank if entry is zero."""
        assert _format_ev_pct(1.0, 0) == ""
        assert _format_ev_pct(5.0, 0.0) == ""

    def test_format_ev_pct_missing_values(self):
        """_format_ev_pct should return blank if expected_value or entry is missing."""
        import numpy as np
        assert _format_ev_pct(None, 100.0) == ""
        assert _format_ev_pct(1.0, None) == ""
        assert _format_ev_pct(float("nan"), 100.0) == ""
        assert _format_ev_pct(1.0, np.nan) == ""


# ============================================================================
# format_table
# ============================================================================

class TestFormatTable:
    def test_format_table_uniform_column_widths_and_alignment(self):
        """format_table should produce lines of uniform length with proper padding."""
        headers = ["name", "value"]
        rows = [
            {"name": "abc", "value": "10"},
            {"name": "x", "value": "9999"},
        ]
        align = ["l", "r"]  # left for name, right for value

        lines = format_table(headers, rows, align)

        # All lines should have the same length (aligned in fixed-width text)
        assert len(lines) > 0
        first_line_len = len(lines[0])
        for line in lines:
            assert len(line) == first_line_len, f"Line length mismatch: {line}"

        # Check that header and separator are present
        assert lines[0].startswith("|")
        assert lines[1].startswith("|")
        assert "-" in lines[1]

        # Check alignment: right-aligned values should have spaces on the left
        assert "abc" in lines[2]  # left-aligned name
        assert "9999" in lines[3] or "9999" in lines[2]  # right-aligned values

    def test_format_table_missing_cells_become_empty_strings(self):
        """format_table should render missing cells as empty strings."""
        headers = ["a", "b"]
        rows = [{"a": "val", "b": None}]
        align = ["l", "l"]

        lines = format_table(headers, rows, align)

        # Should have header, separator, and one data row
        assert len(lines) == 3
        # All lines same length
        first_line_len = len(lines[0])
        for line in lines:
            assert len(line) == first_line_len

    def test_format_table_empty_input(self):
        """format_table should handle empty input."""
        lines = format_table([], [], [])
        assert lines == []


# ============================================================================
# render_report
# ============================================================================

class TestRenderReport:
    def test_only_signal_producing_strategies_in_stats(self):
        stats_rows = [{
            "strategy": "dfa_persistence", "symbol": "BTC-USD", "interval": "1d",
            "backtest_period": "1m", "direction": "LONG", "size": 0.1200,
            "entry": 60800.00, "stop": 58900.00, "target": 63400.00,
            "expected_value": 0.0200,
            "oracle_sharpe": 23.3, "base_sharpe": 30.2,
            "oracle_win_rate": 0.8, "base_win_rate": 1.0,
            "signals_per_week": 0.69,
        }]
        advice_rows = [{
            "expected_value": 0.0200, "entry": 60800.00, "base_win_rate": 1.0,
            "base_signals": 3, "oracle_signals": None,
            "signal": "Strategy dfa_persistence advised **Long** position on BTC-USD ...",
        }]
        failures = ["group assets=A,B interval=1d: boom"]
        skipped = ["ghost_strategy: unknown strategy (not in registry)"]
        ts = datetime(2026, 7, 9, 6, 49)

        report = render_report(stats_rows, advice_rows, failures, skipped, ts)

        assert "# Kairos Signals Report 2026-07-09 0649h" in report
        assert "## Stats" in report
        assert "dfa_persistence" in report
        assert "## Signals" in report
        assert advice_rows[0]["signal"] in report
        assert "## Failures" in report
        assert "boom" in report
        assert "## Skipped" in report
        assert "ghost_strategy" in report
        assert "### Legend" in report
        assert "confidence" not in report

    def test_no_signals_sections_present(self):
        ts = datetime(2026, 7, 9, 6, 49)
        report = render_report([], [], [], [], ts)
        assert "_No strategies produced a signal in this run._" in report
        assert "_No signals generated._" in report
        assert "## Failures" not in report
        assert "## Skipped" not in report

    def test_signals_table_format_has_correct_columns(self):
        """Signals table should have columns: ev_pct, base_win_rate, signals/backtest, signal."""
        advice_rows = [{
            "expected_value": 0.9, "entry": 100.0, "base_win_rate": 0.65,
            "base_signals": 5, "oracle_signals": None,
            "signal": "Strategy test advised **Long** on BTC.",
        }]
        ts = datetime(2026, 7, 9, 6, 49)
        report = render_report([], advice_rows, [], [], ts)

        # Check header row has the correct column names
        assert all(x in report for x in ["ev_pct", "base_win_rate", "signals/backtest", "signal"])
        # confidence column removed from the report entirely
        assert "confidence" not in report
        # Check data is present
        assert "+0.90%" in report  # ev_pct (0.9/100 * 100 = 0.90%)
        assert "5" in report  # signals/backtest

    def test_numeric_cells_rounded_to_2_decimals(self):
        """Stats table numeric cells should format with max 2 decimals."""
        stats_rows = [{
            "strategy": "test", "symbol": "BTC-USD", "interval": "1d",
            "backtest_period": "1m", "direction": "LONG", "size": 0.123456,
            "entry": 60800.456, "stop": 58900.456, "target": 63400.789,
            "expected_value": 0.0123,
            "oracle_sharpe": 23.3456, "base_sharpe": 30.2789,
            "oracle_win_rate": 0.8765, "base_win_rate": 0.9234,
            "signals_per_week": 0.6912,
        }]
        ts = datetime(2026, 7, 9, 6, 49)
        report = render_report(stats_rows, [], [], [], ts)

        # Check that values are formatted to 2 decimals where applicable
        assert "0.12" in report  # size
        assert "60800.46" in report  # entry
        assert "58900.46" in report  # stop
        assert "0.01" in report  # expected_value

    def test_signals_table_ev_pct_computed_correctly(self):
        """Signals table should show ev_pct as percentage of entry price."""
        advice_rows = [{
            "expected_value": 0.9, "entry": 100.0, "base_win_rate": 0.75,
            "base_signals": 10, "oracle_signals": None,
            "signal": "Test signal",
        }]
        ts = datetime(2026, 7, 9, 6, 49)
        report = render_report([], advice_rows, [], [], ts)

        # 0.9 / 100 * 100 = 0.90%
        assert "+0.90%" in report

    def test_signals_table_signals_backtest_fallback_to_oracle(self):
        """signals/backtest should use base_signals, fallback to oracle_signals."""
        advice_rows = [
            {
                "expected_value": 0.9, "entry": 100.0, "base_win_rate": 0.75,
                "base_signals": 5, "oracle_signals": 10,
                "signal": "Test with base",
            },
            {
                "expected_value": 0.5, "entry": 100.0, "base_win_rate": 0.65,
                "base_signals": None, "oracle_signals": 8,
                "signal": "Test fallback",
            },
            {
                "expected_value": 0.3, "entry": 100.0, "base_win_rate": 0.55,
                "base_signals": None, "oracle_signals": None,
                "signal": "Test blank",
            },
        ]
        ts = datetime(2026, 7, 9, 6, 49)
        report = render_report([], advice_rows, [], [], ts)

        # Verify values appear in the report
        lines = report.split("\n")
        signals_section = False
        row_count = 0
        for line in lines:
            if "## Signals" in line:
                signals_section = True
            if signals_section and "|" in line and "Test" in line:
                row_count += 1
                if "Test with base" in line:
                    assert "5" in line
                elif "Test fallback" in line:
                    assert "8" in line
                elif "Test blank" in line:
                    # Should have blank for signals/backtest
                    pass

    def test_render_report_legend_section_present(self):
        """Report should have Legend section with all three column descriptions."""
        advice_rows = [{
            "expected_value": 0.9, "entry": 100.0, "base_win_rate": 0.75,
            "base_signals": 5, "oracle_signals": None,
            "signal": "Test signal",
        }]
        ts = datetime(2026, 7, 9, 6, 49)
        report = render_report([], advice_rows, [], [], ts)

        # Check Legend section exists
        assert "### Legend" in report
        # Check all three column descriptions are present
        assert "ev_pct" in report
        assert "base_win_rate" in report
        assert "signals/backtest" in report
        # confidence removed from table headers and legend
        assert "confidence" not in report
        # Check key descriptions
        assert "expected value of the trade per unit" in report
        assert "fraction of winning trades" in report
        assert "number of signals the strategy generated" in report


# ============================================================================
# load_work_items / group_items
# ============================================================================

VIABILITY_SCHEMA = """
CREATE TABLE viability_report (
    run_id INTEGER,
    strategy_name TEXT,
    assets TEXT,
    asset_class TEXT,
    interval TEXT,
    backtest_period TEXT,
    oracle_sharpe REAL,
    oracle_signals INTEGER,
    oracle_win_rate REAL,
    oracle_avg_pnl_per_trade REAL,
    oracle_run_id INTEGER,
    base_sharpe REAL,
    base_signals INTEGER,
    base_win_rate REAL,
    base_avg_pnl_per_trade REAL,
    base_run_id INTEGER,
    base_model_path TEXT,
    signals_per_week REAL,
    viable INTEGER
);
"""


def _seed_db(tmp_path):
    db_path = os.path.join(tmp_path, "pipeline_results.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(VIABILITY_SCHEMA)
    rows = [
        (1, "dfa_persistence", "BTC-USD,ETH-USD", "crypto", "1d", "1m",
         23.3, 5, 0.8, 0.016, 235, 30.2, 3, 1.0, 0.023, 236, None, 0.69, 1),
        (1, "range_trading", "BTC-USD,ETH-USD", "crypto", "1d", "1m",
         24.1, 8, 0.875, 0.014, 235, 21.7, 3, 1.0, 0.030, 236, None, 0.69, 1),
        (1, "not_viable_strategy", "BTC-USD,ETH-USD", "crypto", "1d", "1m",
         1.0, 1, 0.1, 0.001, 235, 1.0, 1, 0.1, 0.001, 236, None, 0.1, 0),
        (2, "old_run_strategy", "AAPL", "equity", "1d", "1m",
         5.0, 2, 0.5, 0.01, 235, 5.0, 2, 0.5, 0.01, 236, None, 0.3, 1),
    ]
    conn.executemany(
        "INSERT INTO viability_report VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


class TestLoadWorkItemsAndGrouping:
    def test_latest_run_only_and_viable_filter(self, tmp_path):
        db_path = _seed_db(tmp_path)
        conn = sqlite3.connect(db_path)
        rows = load_work_items(conn, intervals=None, include_all=False)
        conn.close()
        # run_id=2 is latest; only rows from run_id=2 should appear.
        names = {r["strategy_name"] for r in rows}
        assert names == {"old_run_strategy"}

    def test_include_all_still_latest_run_only(self, tmp_path):
        db_path = _seed_db(tmp_path)
        conn = sqlite3.connect(db_path)
        rows = load_work_items(conn, intervals=None, include_all=True)
        conn.close()
        names = {r["strategy_name"] for r in rows}
        assert names == {"old_run_strategy"}

    def test_grouping_one_predict_call_per_group(self, tmp_path):
        # Seed a DB where the latest run_id has two strategies sharing one
        # (assets, interval) group.
        db_path = os.path.join(tmp_path, "pipeline_results.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(VIABILITY_SCHEMA)
        rows = [
            (5, "dfa_persistence", "BTC-USD,ETH-USD", "crypto", "1d", "1m",
             23.3, 5, 0.8, 0.016, 235, 30.2, 3, 1.0, 0.023, 236, None, 0.69, 1),
            (5, "range_trading", "BTC-USD,ETH-USD", "crypto", "1d", "1m",
             24.1, 8, 0.875, 0.014, 235, 21.7, 3, 1.0, 0.030, 236, None, 0.69, 1),
        ]
        conn.executemany(
            "INSERT INTO viability_report VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        conn.close()

        conn = sqlite3.connect(db_path)
        work_items = load_work_items(conn, intervals=None, include_all=False)
        conn.close()
        groups = group_items(work_items)
        assert len(groups) == 1
        ((assets, interval), group_rows), = groups.items()
        assert assets == "BTC-USD,ETH-USD"
        assert interval == "1d"
        assert len(group_rows) == 2

    def test_grouping_via_run_calls_predict_once_per_group(self, tmp_path, monkeypatch):
        """End-to-end: run() should call the injected predict_fn exactly once
        per (assets, interval) group, not once per strategy row."""
        import pandas as pd
        import numpy as np

        db_path = os.path.join(tmp_path, "pipeline_results.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(VIABILITY_SCHEMA)
        rows = [
            (7, "dfa_persistence", "BTC-USD", "crypto", "1d", "1m",
             23.3, 5, 0.8, 0.016, 235, 30.2, 3, 1.0, 0.023, 236, None, 0.69, 1),
            (7, "range_trading", "BTC-USD", "crypto", "1d", "1m",
             24.1, 8, 0.875, 0.014, 235, 21.7, 3, 1.0, 0.030, 236, None, 0.69, 1),
        ]
        conn.executemany(
            "INSERT INTO viability_report VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        conn.close()

        idx = pd.date_range("2024-01-01", periods=310, freq="D")
        fake_history = pd.DataFrame({
            "open": np.full(310, 100.0), "high": np.full(310, 101.0),
            "low": np.full(310, 99.0), "close": np.full(310, 100.0),
            "volume": np.full(310, 1e6),
        }, index=idx)

        def fake_fetch_data_raw(symbol, lookback, pred_len=0, min_bars=None):
            return fake_history

        call_count = {"n": 0}

        def fake_predict_fn(assets_dict):
            call_count["n"] += 1
            from kairos_meta import AssetPrediction, KairosDistribution
            frames = [fake_history.iloc[[i]] for i in range(len(fake_history))]
            dist = KairosDistribution(frames[-20:])
            return {
                sym: AssetPrediction(symbol=sym, dist=dist, current_price=100.0, history=fake_history)
                for sym in assets_dict
            }

        import kairos_signals
        monkeypatch.setattr(kairos_signals, "fetch_data_raw", fake_fetch_data_raw, raising=False)
        # Patch the name inside the strategy module's namespace since run()
        # does a local `from kairos_strategies import fetch_data_raw`.
        import kairos_strategies
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)

        out_path = run(
            db_path=db_path, out_dir=str(tmp_path), intervals=None,
            pred_samples=5, include_all=False, predict_fn=fake_predict_fn,
            lookback=300, now=datetime(2026, 7, 9, 6, 49),
        )

        assert call_count["n"] == 1
        assert os.path.exists(out_path)
        assert os.path.basename(out_path) == "kairos_signals_202607090649.md"


# ============================================================================
# build_strategy_index (wrapper unwrapping)
# ============================================================================

class _FakeInner:
    name = "inner_x"

    def generate_signal(self, dist, current_price, history, context):
        return Signal(
            direction=Direction.LONG, size=0.10, entry=current_price,
            stop=current_price * 0.97, target=current_price * 1.05,
            strategy_name="inner_x", confidence=0.7, expected_value=0.02,
        )


class _FakeWrapper:
    name = "fake_wrapper"

    def __init__(self, base_strategy):
        self.base_strategy = base_strategy

    def generate_signal(self, dist, current_price, history, context):
        return self.base_strategy.generate_signal(dist, current_price, history, context)


class TestBuildStrategyIndex:
    def test_inner_name_resolves_to_outermost_wrapper(self):
        inner = _FakeInner()
        wrapper = _FakeWrapper(inner)
        index = build_strategy_index([wrapper])
        assert index["inner_x"] is wrapper
        assert index["fake_wrapper"] is wrapper

    def test_first_seen_exact_match_not_overwritten(self):
        # A bare strategy named "inner_x" registered first must keep its slot;
        # a later wrapper chain containing inner_x must not overwrite it.
        bare = _FakeInner()
        wrapper = _FakeWrapper(_FakeInner())
        index = build_strategy_index([bare, wrapper])
        assert index["inner_x"] is bare

    def test_real_registry_inner_names_resolve(self):
        from kairos_orchestrator import StrategyRegistry, OrchestratorConfig
        # skew is disabled by default; use an empty disabled set so all
        # constructed strategies (including skew) are present.
        strategies = StrategyRegistry.build_all(OrchestratorConfig(disabled_strategies=set()))
        index = build_strategy_index(strategies)
        for name in ("high_low", "vol_target_sizer", "expected_value", "skew"):
            assert name in index, f"{name} missing from strategy index"

    def test_run_resolves_inner_strategy_name(self, tmp_path, monkeypatch):
        """A viability row naming the INNER strategy must produce a signal."""
        import pandas as pd
        import numpy as np

        db_path = os.path.join(tmp_path, "pipeline_results.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(VIABILITY_SCHEMA)
        conn.execute(
            "INSERT INTO viability_report VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (9, "inner_x", "BTC-USD", "crypto", "1d", "1m",
             23.3, 5, 0.8, 0.016, 235, 30.2, 3, 1.0, 0.023, 236, None, 0.69, 1),
        )
        conn.commit()
        conn.close()

        idx = pd.date_range("2024-01-01", periods=310, freq="D")
        fake_history = pd.DataFrame({
            "open": np.full(310, 100.0), "high": np.full(310, 101.0),
            "low": np.full(310, 99.0), "close": np.full(310, 100.0),
            "volume": np.full(310, 1e6),
        }, index=idx)

        import kairos_strategies
        monkeypatch.setattr(
            kairos_strategies, "fetch_data_raw",
            lambda symbol, lookback, pred_len=0, min_bars=None: fake_history,
        )

        def fake_predict_fn(assets_dict):
            from kairos_meta import AssetPrediction, KairosDistribution
            frames = [fake_history.iloc[[i]] for i in range(len(fake_history) - 20, len(fake_history))]
            dist = KairosDistribution(frames)
            return {
                sym: AssetPrediction(symbol=sym, dist=dist, current_price=100.0, history=fake_history)
                for sym in assets_dict
            }

        # Make the registry return only our fake wrapper so the inner name
        # must be resolved through the wrapper chain.
        import kairos_orchestrator
        wrapper = _FakeWrapper(_FakeInner())
        monkeypatch.setattr(
            kairos_orchestrator.StrategyRegistry, "build_all",
            classmethod(lambda cls, config: [wrapper]),
        )
        # Meta filters must not block the synthetic distribution.
        monkeypatch.setattr(
            kairos_orchestrator.KairosOrchestrator, "_apply_meta_filters",
            lambda self, dist, current_price: False,
        )

        out_path = run(
            db_path=db_path, out_dir=str(tmp_path), intervals=None,
            pred_samples=5, include_all=False, predict_fn=fake_predict_fn,
            lookback=300, now=datetime(2026, 7, 9, 7, 0),
        )

        report = open(out_path).read()
        assert "unknown strategy" not in report
        assert "inner_x" in report
        assert "**Long**" in report


# ============================================================================
# Zero-size signal gating (matches backtest sig.size > 0 gate)
# ============================================================================

class _FixedSignalStrategy:
    """Fake strategy returning a preset Signal."""

    def __init__(self, name, signal):
        self.name = name
        self._signal = signal

    def generate_signal(self, dist, current_price, history, context):
        return self._signal


class TestZeroSizeSignalGate:
    def _run_with_strategy(self, tmp_path, monkeypatch, strategy):
        import pandas as pd
        import numpy as np

        db_path = os.path.join(tmp_path, "pipeline_results.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(VIABILITY_SCHEMA)
        conn.execute(
            "INSERT INTO viability_report VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (11, strategy.name, "BTC-USD", "crypto", "1d", "1m",
             23.3, 5, 0.8, 0.016, 235, 30.2, 3, 1.0, 0.023, 236, None, 0.69, 1),
        )
        conn.commit()
        conn.close()

        idx = pd.date_range("2024-01-01", periods=310, freq="D")
        fake_history = pd.DataFrame({
            "open": np.full(310, 100.0), "high": np.full(310, 101.0),
            "low": np.full(310, 99.0), "close": np.full(310, 100.0),
            "volume": np.full(310, 1e6),
        }, index=idx)

        import kairos_strategies
        monkeypatch.setattr(
            kairos_strategies, "fetch_data_raw",
            lambda symbol, lookback, pred_len=0, min_bars=None: fake_history,
        )

        def fake_predict_fn(assets_dict):
            from kairos_meta import AssetPrediction, KairosDistribution
            frames = [fake_history.iloc[[i]] for i in range(len(fake_history) - 20, len(fake_history))]
            dist = KairosDistribution(frames)
            return {
                sym: AssetPrediction(symbol=sym, dist=dist, current_price=100.0, history=fake_history)
                for sym in assets_dict
            }

        import kairos_orchestrator
        monkeypatch.setattr(
            kairos_orchestrator.StrategyRegistry, "build_all",
            classmethod(lambda cls, config: [strategy]),
        )
        monkeypatch.setattr(
            kairos_orchestrator.KairosOrchestrator, "_apply_meta_filters",
            lambda self, dist, current_price: False,
        )

        out_path = run(
            db_path=db_path, out_dir=str(tmp_path), intervals=None,
            pred_samples=5, include_all=False, predict_fn=fake_predict_fn,
            lookback=300, now=datetime(2026, 7, 9, 8, 0),
        )
        return open(out_path).read()

    def test_zero_size_long_dropped_to_skipped(self, tmp_path, monkeypatch):
        sig = Signal(
            direction=Direction.LONG, size=0.0, entry=100.0,
            stop=97.0, target=105.0, strategy_name="zero_kelly",
            confidence=0.7, expected_value=0.02,
        )
        report = self._run_with_strategy(
            tmp_path, monkeypatch, _FixedSignalStrategy("zero_kelly", sig))

        assert "## Skipped" in report
        assert "zero_kelly/BTC-USD: zero-size signal dropped (no Kelly edge)" in report
        # Must not appear as advice or in the stats table.
        assert "advised" not in report
        assert "_No strategies produced a signal in this run._" in report
        assert "_No signals generated._" in report

    def test_flat_zero_size_still_renders_exit_advice(self, tmp_path, monkeypatch):
        sig = Signal(
            direction=Direction.FLAT, size=0.0, entry=100.0,
            stop=0.0, target=0.0, strategy_name="flat_advisor",
            confidence=0.0, expected_value=0.0,
        )
        report = self._run_with_strategy(
            tmp_path, monkeypatch, _FixedSignalStrategy("flat_advisor", sig))

        assert "Strategy flat_advisor advised **Exit/Flat** on BTC-USD." in report
        assert "zero-size signal dropped" not in report
