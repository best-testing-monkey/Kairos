"""Pure-logic checks for scripts/ibkr_instruments.py.

No IB Gateway and no network: only the symbol mapping and the commission-model
inference, which are the two places a silent wrong answer would poison the
whole table.
"""
import importlib.util
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "ibkr_instruments.py")


def _load():
    spec = importlib.util.spec_from_file_location("ibkr_instruments", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ibi = _load()


class TestInferCommissionModel:
    """IBKR charges max(floor, rate * notional); two probes separate the two."""

    def test_both_probes_floor_bound_reports_zero_rate(self):
        # $1.00 at both sizes means the per-unit rate never overtook the floor
        # in this range -- that is a real answer, not missing data.
        floor, rate = ibi.infer_commission_model(1.0, 800.0, 1.0, 25_000.0)
        assert floor == 1.0
        assert rate == 0.0

    def test_rate_bound_large_probe_recovers_both_terms(self):
        # Spot FX: EUR1.7252 floor, 0.20bp above it.
        floor, rate = ibi.infer_commission_model(1.7252, 20_000.0, 2.1565, 100_000.0)
        assert floor == pytest.approx(1.6174, abs=1e-3)
        assert rate == pytest.approx(0.000539, abs=1e-5)

    def test_only_large_probe_gives_rate_but_no_floor(self):
        # A floor cannot be inferred from one point, and guessing 0.0 would
        # read as "this instrument has no minimum commission".
        floor, rate = ibi.infer_commission_model(None, None, 2.51, 25_000.0)
        assert floor is None
        assert rate == pytest.approx(0.01004)

    def test_only_small_probe_gives_floor_but_no_rate(self):
        floor, rate = ibi.infer_commission_model(1.0, 800.0, None, None)
        assert floor == 1.0
        assert rate is None

    def test_no_probes_returns_nothing(self):
        assert ibi.infer_commission_model(None, None, None, None) == (None, None)

    def test_equal_notionals_are_not_treated_as_a_slope(self):
        # Dividing by (n2 - n1) == 0 would raise. Degenerate input must fall
        # through to the single-probe path: a rate from the one usable point,
        # and no floor, since one point cannot establish one.
        floor, rate = ibi.infer_commission_model(1.0, 800.0, 1.0, 800.0)
        assert floor is None
        assert rate == pytest.approx(0.125)


class TestYfToIb:
    """yfinance ticker -> IB contract. A wrong route silently sweeps the wrong
    instrument rather than failing, so each suffix class is pinned."""

    def setup_method(self):
        pytest.importorskip("ib_async")

    def test_crypto_suffix_routes_to_paxos_with_ioc(self):
        contract, cls, tif = ibi.yf_to_ib("BTC-USD")
        assert contract.secType == "CRYPTO"
        assert contract.symbol == "BTC"
        assert contract.exchange == "PAXOS"
        assert cls == "crypto"
        # Crypto rejects DAY outright; the tif is load-bearing, not cosmetic.
        assert tif == "IOC"

    def test_fx_suffix_routes_to_a_forex_pair(self):
        contract, cls, tif = ibi.yf_to_ib("EURUSD=X")
        assert contract.secType == "CASH"
        assert contract.symbol == "EUR"
        assert contract.currency == "USD"
        assert cls == "fx"

    def test_futures_suffix_uses_the_explicit_exchange(self):
        # SMART does not route futures; a missing exchange resolves nothing.
        contract, cls, _ = ibi.yf_to_ib("GC=F")
        assert contract.secType == "FUT"
        assert contract.symbol == "GC"
        assert contract.exchange == "COMEX"
        assert cls == "commodity"

    def test_unmapped_future_returns_no_route(self):
        assert ibi.yf_to_ib("ZZ=F") is None

    def test_bare_index_has_no_tradeable_route(self):
        # IND resolves in the API but cannot be traded; the index CFDs are the
        # tradeable substitute and are probed by --discover instead.
        assert ibi.yf_to_ib("^GSPC") is None

    def test_plain_ticker_routes_to_a_smart_us_stock(self):
        contract, cls, _ = ibi.yf_to_ib("AAPL")
        assert contract.secType == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
        assert cls == "equity"

    def test_international_listing_strips_the_suffix(self):
        contract, cls, _ = ibi.yf_to_ib("0QF.L")
        assert contract.secType == "STK"
        assert contract.symbol == "0QF"
        assert cls == "equity"


class TestNum:
    def test_rejects_ibkr_unset_sentinel(self):
        # IBKR sends DBL_MAX for an unset commission; arithmetic on it produces
        # plausible-looking nonsense rather than an error.
        assert ibi._num(1.7976931348623157e308) is None

    def test_rejects_nan_and_non_numeric(self):
        assert ibi._num(float("nan")) is None
        assert ibi._num(None) is None
        assert ibi._num("") is None

    def test_passes_ordinary_values_through(self):
        assert ibi._num(2.51) == 2.51
        assert ibi._num("3.5") == 3.5


class TestUpsertDoesNotDowngrade:
    """A retry during an IBKR outage must not destroy a good row.

    This is the failure that motivated the guard: the sec-def farm dropped
    mid-sweep, reqContractDetails began timing out, and fully-populated rows
    (contract metadata plus the exact rejection reason) were rewritten as bare
    'error' rows.
    """

    def _row(self, status, note=""):
        return dict(
            yf_symbol="SPY", ib_symbol="SPY", sec_type="STK", exchange="SMART",
            currency="USD", con_id=756733, local_symbol="SPY",
            primary_exchange="ARCA", long_name="SPDR S&P 500", instrument_class="etf",
            broker="IBKR", included_in_universe=1, min_tick=0.01, min_size=1.0,
            size_increment=1.0, ref_price=761.89, ref_price_date="ibkr_delayed",
            small_qty=None, small_notional=None, small_commission=None,
            large_qty=None, large_notional=None, large_commission=None,
            commission_currency=None, commission_min=None, commission_rate_pct=None,
            init_margin_pct=None, maint_margin_pct=None, short_init_margin_pct=None,
            trading_hours=None, time_zone=None, status=status, note=note,
            measured_at="2026-09-02T15:00:00",
        )

    def _conn(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.executescript(ibi.SCHEMA)
        return conn

    def _status(self, conn):
        return conn.execute("SELECT status FROM ibkr_instruments").fetchone()[0]

    def test_transient_error_cannot_replace_a_known_rejection(self):
        conn = self._conn()
        assert ibi.upsert(conn, self._row("no_commission", "201:No Trading Permission"))
        assert ibi.upsert(conn, self._row("error", "details:")) is False
        assert self._status(conn) == "no_commission"
        # The reason text survives, which is the whole point.
        note = conn.execute("SELECT note FROM ibkr_instruments").fetchone()[0]
        assert "No Trading Permission" in note

    def test_error_cannot_replace_a_definitive_no_contract(self):
        conn = self._conn()
        ibi.upsert(conn, self._row("no_contract"))
        assert ibi.upsert(conn, self._row("error")) is False
        assert self._status(conn) == "no_contract"

    def test_better_result_does_replace(self):
        conn = self._conn()
        ibi.upsert(conn, self._row("error"))
        assert ibi.upsert(conn, self._row("ok")) is True
        assert self._status(conn) == "ok"

    def test_equal_rank_still_writes_so_fresh_measurements_land(self):
        conn = self._conn()
        ibi.upsert(conn, self._row("ok", "first"))
        assert ibi.upsert(conn, self._row("ok", "second")) is True
        assert conn.execute("SELECT note FROM ibkr_instruments").fetchone()[0] == "second"

    def test_force_overrides_the_guard(self):
        # Needed if IBKR genuinely delists something that was previously ok.
        conn = self._conn()
        ibi.upsert(conn, self._row("ok"))
        assert ibi.upsert(conn, self._row("error"), force=True) is True
        assert self._status(conn) == "error"

    def test_first_write_always_lands(self):
        conn = self._conn()
        assert ibi.upsert(conn, self._row("error")) is True
