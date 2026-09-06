"""Shared machinery for exchange/broker cost-discovery probes.

A "probe" answers, empirically, for one broker/exchange: what does this
instrument cost to trade (commission, margin), and is it tradeable at all
from our account? `scripts/ibkr_instruments.py` is the reference probe --
this module is what it (and any future exchange's probe) shares, extracted
so a new exchange needs a symbol-resolution + cost-lookup implementation and
nothing else.

See `docs/broker-api-interface.md` for what a probe needs to implement and
why (two capability tiers, two cost-discovery patterns), and
`docs/playbooks/add-exchange-probe.md` for the concrete steps to add one.

This module does NOT cover live order execution (place_order/cancel/
get_positions/get_balance) -- that is Tier 2 in the interface doc, gated on
Phase 4 per `roadmap/phase-5-auto-execution.md`, and nothing implements it
yet.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class InstrumentMeta:
    """What `ExchangeProbe.resolve_instrument()` returns for one symbol."""
    native_symbol: str
    instrument_class: str          # equity, etf, crypto, fx, commodity, ...
    sec_type: str = ""             # the broker's own type tag
    exchange: str = ""
    currency: str = ""
    native_id: str = ""            # conId, ccxt market id, ...
    long_name: str = ""
    min_tick: Optional[float] = None
    price_magnifier: float = 1.0
    min_size: Optional[float] = None
    size_increment: Optional[float] = None
    trading_hours: str = ""
    time_zone: str = ""


@dataclass
class CostModel:
    """What `ExchangeProbe.get_cost_model()` returns.

    `status`/`note` follow the same vocabulary as the DB row's `status`
    column (see `_STATUS_RANK` below): 'ok' | 'no_commission' |
    'no_whatif_crypto' | 'no_price' | 'no_contract' | 'error'. Margin
    fields are None for cash/spot instruments -- most crypto exchanges have
    no margin concept at all, so leaving them unset is the normal case, not
    a missing measurement.
    """
    status: str
    note: str = ""
    commission_min: Optional[float] = None
    commission_rate_pct: Optional[float] = None
    commission_currency: str = ""
    init_margin_pct: Optional[float] = None
    maint_margin_pct: Optional[float] = None
    short_init_margin_pct: Optional[float] = None
    small_qty: Optional[float] = None
    small_notional: Optional[float] = None
    small_commission: Optional[float] = None
    large_qty: Optional[float] = None
    large_notional: Optional[float] = None
    large_commission: Optional[float] = None


@runtime_checkable
class ExchangeProbe(Protocol):
    """What a new exchange/broker cost-discovery probe must implement.

    `scripts/ibkr_instruments.py` is the reference implementation for
    Pattern A (dry-run/whatIf-order cost inference) -- copy its shape, not
    its guts, for another whatIf-style broker. For Pattern B (an exchange
    that exposes its fee schedule directly, which is every ccxt-supported
    crypto exchange), implement this Protocol and drive it with
    `sweep_simple()` below -- see `FakeExchangeProbe` in
    tests/unit/test_exchange_probe.py for the minimal shape that proves it.
    """
    broker: str

    def connect(self, args) -> None: ...
    def disconnect(self) -> None: ...
    def resolve_instrument(self, symbol: str) -> Optional[InstrumentMeta]: ...
    def get_reference_price(self, instrument: InstrumentMeta) -> Optional[float]: ...
    def get_cost_model(self, instrument: InstrumentMeta, ref_price: float,
                        base_currency: str) -> CostModel: ...
    def get_fx_rate(self, currency: str, base: str) -> Optional[float]: ...


# Generic per-exchange schema for a new probe (Pattern B). IBKR keeps its own
# literal SCHEMA in ibkr_instruments.py unchanged -- it already has ~190 rows
# of real measured data and a different (but equivalent) column naming
# (`ib_symbol` instead of `symbol`); migrating it wasn't worth the risk for
# this pass. A new exchange's table starts clean with this instead.
SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS instruments (
    symbol                TEXT NOT NULL,
    sec_type              TEXT NOT NULL,
    exchange              TEXT,
    currency              TEXT,
    native_id             TEXT,
    long_name             TEXT,
    instrument_class      TEXT NOT NULL,
    broker                TEXT NOT NULL,
    included_in_universe  INTEGER NOT NULL,
    min_tick              REAL,
    price_magnifier       REAL,
    min_size              REAL,
    size_increment        REAL,
    ref_price             REAL,
    ref_price_date        TEXT,
    base_currency         TEXT,
    fx_rate_to_base       REAL,
    small_qty             REAL,
    small_notional        REAL,
    small_commission      REAL,
    large_qty             REAL,
    large_notional        REAL,
    large_commission      REAL,
    commission_currency   TEXT,
    commission_min        REAL,
    commission_rate_pct   REAL,
    init_margin_pct       REAL,
    maint_margin_pct      REAL,
    short_init_margin_pct REAL,
    trading_hours         TEXT,
    time_zone             TEXT,
    status                TEXT NOT NULL,
    note                  TEXT,
    measured_at           TEXT NOT NULL,
    PRIMARY KEY (broker, symbol, sec_type, exchange, currency)
);
"""

TEMPLATE_KEY_COLS = ("broker", "symbol", "sec_type", "exchange", "currency")


def _num(value):
    """Reject non-finite/sentinel values rather than letting them poison
    arithmetic (IBKR sends DBL_MAX for 'unset'; ccxt can hand back None or
    NaN for a fee it doesn't know)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (f != f or abs(f) > 1e100) else f


def infer_commission_model(c1, n1, c2, n2):
    """Split two (commission, notional) probes into (floor, marginal rate %).

    Only needed for Pattern A (whatIf-style probing, no direct fee-schedule
    endpoint) -- see scripts/ibkr_instruments.py's docstring of the same
    name for the full derivation. A Pattern B exchange gets floor/rate
    straight from its fee-schedule call and does not need this.
    """
    if c1 is not None and c2 is not None and n1 and n2 and n2 > n1:
        if abs(c2 - c1) < 1e-6:
            return c1, 0.0
        rate = (c2 - c1) / (n2 - n1)
        return max(0.0, c1 - rate * n1), 100.0 * rate
    if c2 is not None and n2:
        return None, 100.0 * c2 / n2
    if c1 is not None and n1:
        return c1, None
    return None, None


# How informative a row is. A retry that learns less than what is already
# stored must not overwrite it -- see ibkr_instruments.py's own note on the
# outage that motivated this (a transient failure rewrote fully-populated
# rows as bare 'error' rows).
_STATUS_RANK = {
    "ok": 4,
    "no_commission": 3,      # resolved, cost lookup rejected/unsupported
    "no_whatif_crypto": 3,   # ditto, and terminal (no margin product exists)
    "no_price": 2,           # resolved, no reference price available
    "no_contract": 1,        # definitive: this exchange does not list it
    "error": 0,              # transient or unknown
}

# Statuses no retry can improve -- skipped even under --force, which is meant
# to re-measure things that could have changed, not re-ask a settled question.
_STRUCTURALLY_TERMINAL = ("no_contract", "no_whatif_crypto")


def connect_db(db_path: str, schema_sql: str, table: str,
                extra_columns: Optional[dict] = None) -> sqlite3.Connection:
    """Open (creating if needed) a probe DB, applying any additive column
    migrations. `extra_columns` is `{col_name: sql_type}` for columns added
    to `schema_sql` after the table may already exist on disk."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)
    if extra_columns:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, decl in extra_columns.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    conn.commit()
    return conn


def upsert(conn, row: dict, table: str, key_cols, force: bool = False) -> bool:
    """Write a row unless it would replace a strictly more informative one
    (see `_STATUS_RANK`). Returns whether the write happened."""
    if not force:
        placeholders = " AND ".join(f"{c}=?" for c in key_cols)
        prev = conn.execute(
            f"SELECT status FROM {table} WHERE {placeholders}",
            [row[c] for c in key_cols],
        ).fetchone()
        if prev and _STATUS_RANK.get(row["status"], 0) < _STATUS_RANK.get(prev[0], 0):
            return False
    cols = list(row)
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        [row[c] for c in cols],
    )
    conn.commit()
    return True


def structurally_terminal(conn, table: str, key_cols) -> set:
    cols = ",".join(key_cols)
    placeholders = ",".join("?" * len(_STRUCTURALLY_TERMINAL))
    return {
        tuple(r) for r in conn.execute(
            f"SELECT {cols} FROM {table} WHERE status IN ({placeholders})",
            _STRUCTURALLY_TERMINAL,
        )
    }


def already_done(conn, table: str, key_cols) -> set:
    cols = ",".join(key_cols)
    return {
        tuple(r) for r in conn.execute(
            f"SELECT {cols} FROM {table} WHERE status IN "
            f"('ok','no_contract','no_whatif_crypto')"
        )
    }


def report(conn, table: str) -> None:
    """Summarise a SCHEMA_TEMPLATE-shaped probe table: what trades, what it
    costs, what does not. IBKR keeps its own richer `report()` (it has an
    extra FX-migration warning specific to its own history) -- this is the
    generic version for a new exchange's table."""
    print("=== TRADEABLE (status='ok') ===")
    print(f"  {'class':12s} {'n':>3s}  {'min comm':>9s} {'rate %':>8s} "
          f"{'margin long':>13s} {'margin short':>13s}")
    for cls, n, cmin, rate, lo, hi, sh in conn.execute(
        f"SELECT instrument_class, COUNT(*), ROUND(AVG(commission_min),4), "
        f"ROUND(AVG(commission_rate_pct),5), ROUND(MIN(init_margin_pct),2), "
        f"ROUND(MAX(init_margin_pct),2), ROUND(AVG(short_init_margin_pct),2) "
        f"FROM {table} WHERE status='ok' GROUP BY 1 ORDER BY 1"
    ):
        rng = f"{lo}-{hi}%" if lo is not None and lo != hi else (f"{lo}%" if lo is not None else "-")
        print(f"  {cls:12s} {n:>3d}  {str(cmin):>9s} {str(rate):>8s} {rng:>13s} "
              f"{str(sh) + '%' if sh is not None else '-':>13s}")

    print("=== NOT TRADEABLE ===")
    for cls, status, n in conn.execute(
        f"SELECT instrument_class, status, COUNT(*) FROM {table} "
        f"WHERE status!='ok' GROUP BY 1,2 ORDER BY 3 DESC"
    ):
        print(f"  {cls:12s} {status:18s} n={n}")

    total, inuni = conn.execute(
        f"SELECT COUNT(*), SUM(included_in_universe) FROM {table}"
    ).fetchone()
    ok = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE status='ok'").fetchone()[0]
    print(f"=== {total} rows | {inuni} in universe, {total - (inuni or 0)} discovered "
          f"| {ok} tradeable ===")


def sweep_simple(probe: "ExchangeProbe", conn, table: str, symbols, *,
                  base_currency: str = "EUR", in_universe: bool = True,
                  force: bool = False) -> None:
    """Generic sweep loop for Pattern B (direct fee-schedule lookup, e.g.
    ccxt's `fetch_trading_fee` -- no dual-probe floor/rate inference
    needed). Requires a `table` created from `SCHEMA_TEMPLATE`.

    For Pattern A (dry-run/whatIf-order probing, IBKR's mechanism), write
    your own loop that calls `infer_commission_model` directly -- see
    scripts/ibkr_instruments.py's `sweep_one` for that shape. Forcing both
    patterns through one generic loop was tried and dropped: whatIf's
    two-probe dance and a one-call fee lookup don't share enough real
    structure to be worth the abstraction.
    """
    done = (structurally_terminal(conn, table, TEMPLATE_KEY_COLS) if force
            else already_done(conn, table, TEMPLATE_KEY_COLS))
    now = dt.datetime.now().isoformat(timespec="seconds")

    for i, symbol in enumerate(symbols, 1):
        inst = probe.resolve_instrument(symbol)
        if inst is None:
            row = _blank_row(symbol, probe.broker, in_universe, "no_contract", now)
            upsert(conn, row, table, TEMPLATE_KEY_COLS, force=force)
            print(f"  [{i}/{len(symbols)}] {symbol:16s} no_contract", flush=True)
            continue

        key = (probe.broker, inst.native_symbol, inst.sec_type, inst.exchange, inst.currency)
        if key in done:
            continue

        price = probe.get_reference_price(inst)
        if not price:
            row = _blank_row(symbol, probe.broker, in_universe, "no_price", now)
            row.update(sec_type=inst.sec_type, exchange=inst.exchange, currency=inst.currency,
                       instrument_class=inst.instrument_class, symbol=inst.native_symbol)
            upsert(conn, row, table, TEMPLATE_KEY_COLS, force=force)
            print(f"  [{i}/{len(symbols)}] {symbol:16s} no_price", flush=True)
            continue

        cost = probe.get_cost_model(inst, price, base_currency)
        fx = probe.get_fx_rate(inst.currency, base_currency)

        row = dict(
            symbol=inst.native_symbol, sec_type=inst.sec_type, exchange=inst.exchange,
            currency=inst.currency, broker=probe.broker, native_id=inst.native_id,
            long_name=inst.long_name, instrument_class=inst.instrument_class,
            included_in_universe=int(in_universe), min_tick=inst.min_tick,
            price_magnifier=inst.price_magnifier, min_size=inst.min_size,
            size_increment=inst.size_increment, ref_price=price, ref_price_date=now,
            base_currency=base_currency, fx_rate_to_base=fx,
            small_qty=cost.small_qty, small_notional=cost.small_notional,
            small_commission=cost.small_commission,
            large_qty=cost.large_qty, large_notional=cost.large_notional,
            large_commission=cost.large_commission,
            commission_currency=cost.commission_currency,
            commission_min=cost.commission_min, commission_rate_pct=cost.commission_rate_pct,
            init_margin_pct=cost.init_margin_pct, maint_margin_pct=cost.maint_margin_pct,
            short_init_margin_pct=cost.short_init_margin_pct,
            trading_hours=inst.trading_hours, time_zone=inst.time_zone,
            status=cost.status, note=cost.note, measured_at=now,
        )
        upsert(conn, row, table, TEMPLATE_KEY_COLS, force=force)
        c = row["commission_min"]
        print(f"  [{i}/{len(symbols)}] {symbol:16s} {row['status']:14s} "
              f"comm={'-' if c is None else format(c, '.4f')}", flush=True)


def _blank_row(symbol, broker, in_universe, status, now) -> dict:
    return dict(
        symbol=symbol, sec_type="", exchange="", currency="", broker=broker,
        native_id="", long_name="", instrument_class="",
        included_in_universe=int(in_universe), min_tick=None, price_magnifier=None,
        min_size=None, size_increment=None, ref_price=None, ref_price_date=None,
        base_currency=None, fx_rate_to_base=None,
        small_qty=None, small_notional=None, small_commission=None,
        large_qty=None, large_notional=None, large_commission=None,
        commission_currency=None, commission_min=None, commission_rate_pct=None,
        init_margin_pct=None, maint_margin_pct=None, short_init_margin_pct=None,
        trading_hours=None, time_zone=None, status=status, note="", measured_at=now,
    )
