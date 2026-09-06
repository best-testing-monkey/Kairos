"""Tests for scripts/exchange_probe.py -- the shared cost-probe machinery.

`infer_commission_model`/`_num` are pinned by test_ibkr_instruments.py,
which imports them through ibkr_instruments.py's re-export of this same
module; not duplicated here.

The real point of this file is `TestSweepSimpleEndToEnd`: a fake exchange
implementing nothing but the `ExchangeProbe` Protocol, run through the
actual shared sweep/upsert/report pipeline. This is the proof that a new
exchange probe (Bitvavo, Kraken, ...) plugs into the shared machinery with
no changes to it -- see docs/playbooks/add-exchange-probe.md.
"""
import os
import sqlite3
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import exchange_probe as ep


class FakeExchangeProbe:
    """Minimal Pattern B (direct fee-schedule) probe: three symbols, one of
    each terminal outcome a real exchange can hand back."""
    broker = "FakeX"

    def __init__(self):
        self.connected = False

    def connect(self, args):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def resolve_instrument(self, symbol):
        if symbol == "NOPE":
            return None  # not listed on this exchange
        cls = "crypto" if symbol.endswith("-USD") else "equity"
        return ep.InstrumentMeta(
            native_symbol=symbol.replace("-USD", ""), instrument_class=cls,
            sec_type="spot", exchange="FAKE", currency="USD",
            native_id=f"id-{symbol}", min_tick=0.01, size_increment=0.0001,
        )

    def get_reference_price(self, instrument):
        if instrument.native_symbol == "NOPRICE":
            return None
        return 100.0

    def get_cost_model(self, instrument, ref_price, base_currency):
        return ep.CostModel(status="ok", commission_min=0.0,
                             commission_rate_pct=0.25, commission_currency="USD")

    def get_fx_rate(self, currency, base):
        return 1.0 if currency == base else 0.92


def test_fake_exchange_satisfies_the_protocol():
    assert isinstance(FakeExchangeProbe(), ep.ExchangeProbe)


class TestSweepSimpleEndToEnd:
    def _conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(ep.SCHEMA_TEMPLATE)
        return conn

    def test_full_sweep_writes_ok_missing_and_no_price_rows(self):
        conn = self._conn()
        probe = FakeExchangeProbe()
        ep.sweep_simple(probe, conn, "instruments", ["BTC-USD", "NOPE", "NOPRICE"],
                         base_currency="USD")

        rows = {r[0]: r[1] for r in conn.execute("SELECT symbol, status FROM instruments")}
        assert rows["BTC"] == "ok"
        assert rows["NOPE"] == "no_contract"
        assert rows["NOPRICE"] == "no_price"

        ok_row = conn.execute(
            "SELECT commission_rate_pct, fx_rate_to_base, broker FROM instruments "
            "WHERE symbol='BTC'"
        ).fetchone()
        assert ok_row == (0.25, 1.0, "FakeX")

    def test_resumable_second_sweep_skips_already_ok_symbols(self):
        conn = self._conn()
        probe = FakeExchangeProbe()
        ep.sweep_simple(probe, conn, "instruments", ["BTC-USD"])
        # A second probe object whose get_cost_model would return something
        # different -- if the sweep re-visited it, the row would change.
        probe2 = FakeExchangeProbe()
        probe2.get_cost_model = lambda *a: ep.CostModel(status="ok", commission_rate_pct=9.99)
        ep.sweep_simple(probe2, conn, "instruments", ["BTC-USD"])
        rate = conn.execute(
            "SELECT commission_rate_pct FROM instruments WHERE symbol='BTC'"
        ).fetchone()[0]
        assert rate == 0.25

    def test_report_runs_against_a_freshly_swept_table(self, capsys):
        conn = self._conn()
        ep.sweep_simple(FakeExchangeProbe(), conn, "instruments", ["BTC-USD", "NOPE"])
        ep.report(conn, "instruments")
        out = capsys.readouterr().out
        assert "TRADEABLE" in out
        assert "no_contract" in out


class TestUpsertRankGuardIsBrokerAgnostic:
    """Same guard as ibkr_instruments.py's TestUpsertDoesNotDowngrade, but
    exercised through the generic (table, key_cols) signature directly."""

    def _conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(ep.SCHEMA_TEMPLATE)
        return conn

    def _row(self, status):
        return dict(
            symbol="BTC", sec_type="spot", exchange="FAKE", currency="USD",
            broker="FakeX", native_id="1", long_name="", instrument_class="crypto",
            included_in_universe=1, min_tick=None, price_magnifier=None, min_size=None,
            size_increment=None, ref_price=100.0, ref_price_date="2026-01-01",
            base_currency="USD", fx_rate_to_base=1.0,
            small_qty=None, small_notional=None, small_commission=None,
            large_qty=None, large_notional=None, large_commission=None,
            commission_currency="USD", commission_min=0.0, commission_rate_pct=0.25,
            init_margin_pct=None, maint_margin_pct=None, short_init_margin_pct=None,
            trading_hours=None, time_zone=None, status=status, note="",
            measured_at="2026-01-01T00:00:00",
        )

    def test_transient_error_cannot_replace_ok(self):
        conn = self._conn()
        assert ep.upsert(conn, self._row("ok"), "instruments", ep.TEMPLATE_KEY_COLS)
        assert not ep.upsert(conn, self._row("error"), "instruments", ep.TEMPLATE_KEY_COLS)
        status = conn.execute("SELECT status FROM instruments").fetchone()[0]
        assert status == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
