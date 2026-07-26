import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import json
from datetime import datetime

import pytest

from kairos_papertrade import (
    parse_report_effective_dt,
    generate_and_dedupe_reports,
    map_instrument_type,
    compute_pct_profit_per_trade,
    write_json_report,
)


# ============================================================================
# parse_report_effective_dt
# ============================================================================

class TestParseReportEffectiveDt:
    def test_parses_valid_header(self, tmp_path):
        report = tmp_path / "kairos_signals_202607192052.md"
        report.write_text(
            "# Kairos Signals Report 2026-07-19 2052h\n\n_Filters: min ev_pct 0.10%_\n"
        )
        result = parse_report_effective_dt(str(report))
        assert result == datetime(2026, 7, 19, 20, 52)

    def test_parses_midnight_header(self, tmp_path):
        report = tmp_path / "kairos_signals_202607110000.md"
        report.write_text("# Kairos Signals Report 2026-07-11 0000h\n")
        result = parse_report_effective_dt(str(report))
        assert result == datetime(2026, 7, 11, 0, 0)

    def test_never_falls_back_to_filename(self, tmp_path):
        # Filename says one date, but header (the source of truth) is
        # missing/garbled -- must raise, not silently trust the filename.
        report = tmp_path / "kairos_signals_202601010000.md"
        report.write_text("Some unrelated first line\n\n# Kairos Signals Report 2026-07-19 2052h\n")
        with pytest.raises(ValueError):
            parse_report_effective_dt(str(report))

    def test_rejects_malformed_header(self, tmp_path):
        report = tmp_path / "bad.md"
        report.write_text("# Kairos Signals Report not-a-date\n")
        with pytest.raises(ValueError):
            parse_report_effective_dt(str(report))

    def test_rejects_empty_file(self, tmp_path):
        report = tmp_path / "empty.md"
        report.write_text("")
        with pytest.raises(ValueError):
            parse_report_effective_dt(str(report))


# ============================================================================
# generate_and_dedupe_reports
# ============================================================================

class TestGenerateAndDedupeReports:
    def test_dedupes_by_effective_dt_first_seen_wins(self, tmp_path, monkeypatch):
        # Simulate a weekend: several distinct `now` values (iter_now) all
        # resolve to the SAME last-closed-bar effective_dt in their report
        # header (e.g. Sat/Sun/Mon-morning all show Friday's close).
        calls = []

        def fake_run(now, intervals, return_rows, **kwargs):
            calls.append(now)
            # First two calls collapse to the same effective_dt (weekend
            # dup); the third is a distinct, older effective_dt.
            if len(calls) <= 2:
                effective = "2026-07-17 0000h"
            else:
                effective = "2026-07-16 0000h"
            report_path = tmp_path / f"report_{len(calls)}.md"
            report_path.write_text(f"# Kairos Signals Report {effective}\n")
            stats_rows = [{"call": len(calls)}]
            advice_rows = [{"call": len(calls)}]
            return str(report_path), stats_rows, advice_rows

        import kairos_papertrade as kp
        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)

        base_now = datetime(2026, 7, 19, 0, 0)
        result = generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={})

        # 0.1 months * 30.44 ~= 3.044 -> round() -> 3 iterations
        assert len(calls) == 3
        # Only 2 distinct effective_dts should survive de-dup.
        assert len(result) == 2
        # Sorted oldest-first.
        assert result[0][0] < result[1][0]
        assert result[0][0] == datetime(2026, 7, 16, 0, 0)
        assert result[1][0] == datetime(2026, 7, 17, 0, 0)
        # First-seen wins: effective_dt 2026-07-17 should keep call #1's rows.
        assert result[1][1] == [{"call": 1}]

    def test_returns_empty_list_when_no_iterations(self, monkeypatch):
        import kairos_papertrade as kp

        def fake_run(now, intervals, return_rows, **kwargs):
            raise AssertionError("run() should never be called for 0 iterations")

        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)
        base_now = datetime(2026, 7, 19, 0, 0)
        result = generate_and_dedupe_reports(base_now, "1d", months_back=0.0, run_kwargs={})
        assert result == []


# ============================================================================
# map_instrument_type
# ============================================================================

class TestMapInstrumentType:
    def test_futures_ticker_is_cfd(self):
        assert map_instrument_type({"ticker": "CL=F", "direction": "long"}) == "cfd"
        assert map_instrument_type({"ticker": "NG=F", "direction": "long"}) == "cfd"

    def test_crypto_ticker_is_cfd(self):
        assert map_instrument_type({"ticker": "WIF-USD", "direction": "long"}) == "cfd"
        assert map_instrument_type({"ticker": "UNI7083-USD", "direction": "long"}) == "cfd"

    def test_forex_ticker_is_cfd(self):
        assert map_instrument_type({"ticker": "EURUSD=X", "direction": "long"}) == "cfd"

    def test_plain_equity_ticker_is_stock(self):
        assert map_instrument_type({"ticker": "AAPL", "direction": "long"}) == "stock"
        assert map_instrument_type({"ticker": "NFLX", "direction": "long"}) == "stock"

    def test_short_direction_is_always_cfd(self):
        assert map_instrument_type({"ticker": "AAPL", "direction": "short"}) == "cfd"

    def test_accepts_attribute_object_not_just_dict(self):
        from types import SimpleNamespace
        cand = SimpleNamespace(ticker="AAPL", direction="long")
        assert map_instrument_type(cand) == "stock"
        cand_short = SimpleNamespace(ticker="TSLA", direction="short")
        assert map_instrument_type(cand_short) == "cfd"


# ============================================================================
# compute_pct_profit_per_trade
# ============================================================================

class TestComputePctProfitPerTrade:
    def test_computes_mean_pct_across_positions(self):
        # Two fake closed positions: +10% and -5% (of entry_price*quantity).
        positions = [
            {"realized_pnl": 100.0, "entry_price": 10.0, "quantity": 100.0},  # 100/1000 = 10%
            {"realized_pnl": -25.0, "entry_price": 5.0, "quantity": 100.0},   # -25/500 = -5%
        ]
        result = compute_pct_profit_per_trade(positions)
        assert result == pytest.approx((10.0 + -5.0) / 2)

    def test_ignores_positions_missing_fields(self):
        positions = [
            {"realized_pnl": None, "entry_price": 10.0, "quantity": 100.0},
            {"realized_pnl": 100.0, "entry_price": 10.0, "quantity": 100.0},
        ]
        result = compute_pct_profit_per_trade(positions)
        assert result == pytest.approx(10.0)

    def test_returns_none_for_no_positions(self):
        assert compute_pct_profit_per_trade([]) is None

    def test_returns_none_when_all_positions_uncomputable(self):
        positions = [{"realized_pnl": None, "entry_price": 0.0, "quantity": 0.0}]
        assert compute_pct_profit_per_trade(positions) is None

    def test_accepts_attribute_objects(self):
        from types import SimpleNamespace
        positions = [
            SimpleNamespace(realized_pnl=50.0, entry_price=10.0, quantity=50.0),  # 50/500=10%
        ]
        assert compute_pct_profit_per_trade(positions) == pytest.approx(10.0)


# ============================================================================
# write_json_report
# ============================================================================

class TestWriteJsonReport:
    def test_writes_expected_shape(self, tmp_path):
        metrics = {
            "total_profit_eur": 12.5,
            "pct_profit": 6.25,
            "pct_profit_per_trade": 1.5,
            "pct_max_drawdown": 3.2,
            "sharpe": 0.8,
            "num_trades": 4,
        }
        meta = {"start": "2026-07-01T00:00:00", "end": "2026-07-19T00:00:00", "interval": "1d"}
        out_path = tmp_path / "report.json"

        result_path = write_json_report(metrics, meta, out_path)

        assert os.path.exists(result_path)
        with open(result_path) as f:
            payload = json.load(f)

        for key in ("total_profit_eur", "pct_profit", "pct_profit_per_trade",
                    "pct_max_drawdown", "sharpe", "num_trades"):
            assert key in payload
            assert payload[key] == metrics[key]

        assert "meta" in payload
        assert payload["meta"] == meta

    def test_creates_parent_directories(self, tmp_path):
        out_path = tmp_path / "nested" / "dir" / "report.json"
        write_json_report({"num_trades": 0}, {}, out_path)
        assert out_path.exists()
