#!/usr/bin/env python
"""Build a per-symbol IBKR instrument + commission table from the live Gateway.

Writes `data/ibkr_instruments.db` (table `ibkr_instruments`). Two modes:

    --universe   sweep CANDIDATE_UNIVERSE (included_in_universe=1)
    --discover   probe extra tradeable instruments IBKR offers that the
                 universe doesn't name yet (included_in_universe=0)

Commission comes from `whatIfOrder`, which is non-transmitting -- nothing is
ever placed. Probe orders are sized off a delayed IBKR snapshot, falling back
to the local `data/yfd_prices.db` mirror when a contract has no market-data
entitlement. The mirror alone is not enough: its coverage is patchy and it ran
8 days stale / 5.3% off on AAPL, which both risks an off-market rejection and
skews every margin percentage (those divide by qty*price while IBKR uses the
real price).

Two hard-won constraints shape the probing (see docs/ibkr-cost-discovery.md):

* whatIf only returns a commission for notionals roughly in [$249, $65_200].
  Outside that band the field is silently UNSET with no warning emitted, so
  every probe is sized into the band from the symbol's own last close.
* whatIf resolves to an empty list instead of an OrderState whenever ANY
  warning fires -- an off-market limit price is enough. Limits are set at the
  last close, and `tif` is always set explicitly ('IOC' for crypto, which
  rejects DAY).

Resumable: a symbol already recorded with status='ok' is skipped, so an
interrupted sweep continues where it stopped.

    uv run --with ib_async scripts/ibkr_instruments.py --universe --pace 1.5
    uv run --with ib_async scripts/ibkr_instruments.py --discover
    uv run --with ib_async scripts/ibkr_instruments.py --report
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "strategy"))

DB_PATH = os.path.join(REPO_ROOT, "data", "ibkr_instruments.db")
PRICE_DB = os.path.join(REPO_ROOT, "data", "yfd_prices.db")

# whatIf returns a commission only inside this notional band (measured
# 2026-09-02: $249 OK / $69 UNSET at the bottom, $65_200 OK / $81_500 UNSET at
# the top). Probe at two points well inside it so `min` and `rate` separate:
# the small probe is floor-bound, the large one rate-bound.
PROBE_SMALL_USD = 800.0
PROBE_LARGE_USD = 25_000.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS ibkr_instruments (
    yf_symbol            TEXT,
    ib_symbol            TEXT NOT NULL,
    sec_type             TEXT NOT NULL,
    exchange             TEXT,
    currency             TEXT,
    con_id               INTEGER,
    local_symbol         TEXT,
    primary_exchange     TEXT,
    long_name            TEXT,
    instrument_class     TEXT NOT NULL,
    broker               TEXT NOT NULL DEFAULT 'IBKR',
    included_in_universe INTEGER NOT NULL,
    min_tick             REAL,
    min_size             REAL,
    size_increment       REAL,
    ref_price            REAL,
    ref_price_date       TEXT,
    small_qty            REAL,
    small_notional       REAL,
    small_commission     REAL,
    large_qty            REAL,
    large_notional       REAL,
    large_commission     REAL,
    commission_currency  TEXT,
    commission_min       REAL,
    commission_rate_pct  REAL,
    init_margin_pct      REAL,
    maint_margin_pct     REAL,
    short_init_margin_pct REAL,
    trading_hours        TEXT,
    time_zone            TEXT,
    status               TEXT NOT NULL,
    note                 TEXT,
    measured_at          TEXT NOT NULL,
    PRIMARY KEY (ib_symbol, sec_type, exchange, currency)
);
"""

# yfinance =F tickers -> (IB symbol, exchange). Futures need an explicit
# exchange; SMART does not route them.
FUTURES_MAP = {
    "GC=F": ("GC", "COMEX"), "SI=F": ("SI", "COMEX"), "HG=F": ("HG", "COMEX"),
    "CL=F": ("CL", "NYMEX"), "NG=F": ("NG", "NYMEX"),
    "ZC=F": ("ZC", "CBOT"), "ZW=F": ("ZW", "CBOT"), "ZS=F": ("ZS", "CBOT"),
}

# Instruments IBKR offers that CANDIDATE_UNIVERSE never names. Probed by
# --discover. (label, secType, symbol, exchange, currency, instrument_class)
DISCOVER_TARGETS = [
    # Index CFDs -- the leveraged route to an index, since IND is data-only.
    ("CFD", "IBUS500", "SMART", "USD", "cfd_index"),
    ("CFD", "IBUS30", "SMART", "USD", "cfd_index"),
    ("CFD", "IBUST100", "SMART", "USD", "cfd_index"),
    ("CFD", "IBDE40", "SMART", "EUR", "cfd_index"),
    ("CFD", "IBGB100", "SMART", "GBP", "cfd_index"),
    ("CFD", "IBFR40", "SMART", "EUR", "cfd_index"),
    ("CFD", "IBEU50", "SMART", "EUR", "cfd_index"),
    ("CFD", "IBJP225", "SMART", "JPY", "cfd_index"),
    ("CFD", "IBAU200", "SMART", "AUD", "cfd_index"),
    ("CFD", "IBES35", "SMART", "EUR", "cfd_index"),
    ("CFD", "IBCH20", "SMART", "CHF", "cfd_index"),
    ("CFD", "IBNL25", "SMART", "EUR", "cfd_index"),
    ("CFD", "IBHK50", "SMART", "HKD", "cfd_index"),
    # Metal CFDs / spot metals.
    ("CFD", "XAUUSD", "SMART", "USD", "cfd_metal"),
    ("CFD", "XAGUSD", "SMART", "USD", "cfd_metal"),
    ("CMDTY", "XAUUSD", "SMART", "USD", "metal"),
    ("CMDTY", "XAGUSD", "SMART", "USD", "metal"),
    # FX CFDs (distinct from IDEALPRO spot -- different margin).
    ("CFD", "EUR", "SMART", "USD", "cfd_fx"),
    ("CFD", "GBP", "SMART", "USD", "cfd_fx"),
    ("CFD", "AUD", "SMART", "USD", "cfd_fx"),
    ("CFD", "USD", "SMART", "JPY", "cfd_fx"),
]

# Spot FX pairs on IDEALPRO beyond whatever the universe already lists.
DISCOVER_FX = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "GBPJPY", "AUDJPY",
    "CADJPY", "CHFJPY", "NZDJPY", "AUDNZD", "GBPAUD", "GBPCAD", "USDSEK",
    "USDNOK", "USDMXN", "USDZAR", "USDHKD", "USDSGD", "EURSEK", "EURNOK",
]

# Crypto on PAXOS. Cannot be whatIf'd (the Paxos segment carries no margin, so
# IBKR returns "MARGIN CALCULATION IS NOT SUPPORTED"), but contract metadata
# and tradeability still record fine.
DISCOVER_CRYPTO = [
    "BTC", "ETH", "LTC", "BCH", "LINK", "UNI", "AAVE", "CRV", "MATIC",
    "DOGE", "SOL", "ADA", "AVAX", "DOT", "XRP", "SHIB", "PAXG",
]


def connect_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def load_prices() -> dict:
    """symbol -> (close, date) from the most recent daily bar in the mirror."""
    if not os.path.exists(PRICE_DB):
        return {}
    conn = sqlite3.connect(f"file:{PRICE_DB}?mode=ro", uri=True)
    # SQLite bare-column rule: with MAX(date) aggregated, `close` comes from
    # the row that supplied the max. That is the latest close per ticker.
    rows = conn.execute(
        "SELECT ticker, close, MAX(date) FROM prices WHERE interval_minutes = 1440 "
        "AND close IS NOT NULL AND close > 0 GROUP BY ticker"
    ).fetchall()
    conn.close()
    return {t: (c, d) for t, c, d in rows}


def yf_to_ib(yf_symbol: str):
    """Map a yfinance ticker to (Contract, instrument_class, tif).

    Returns None for tickers with no tradeable IB route (bare indices).
    """
    from ib_async import CFD, Contract, Crypto, Forex, Future, Stock

    s = yf_symbol.strip()
    if s.endswith("-USD"):
        return Crypto(s[:-4], "PAXOS", "USD"), "crypto", "IOC"
    if s.endswith("=X"):
        pair = s[:-2]
        if len(pair) == 6:
            return Forex(pair), "fx", "DAY"
        # yfinance writes some crosses as 'EURUSD=X' and others as 'EUR=X'
        return Forex(f"{pair}USD" if len(pair) == 3 else pair), "fx", "DAY"
    if s.endswith("=F"):
        mapped = FUTURES_MAP.get(s)
        if not mapped:
            return None
        sym, exch = mapped
        return Future(sym, exchange=exch), "commodity", "DAY"
    if s.startswith("^"):
        # IND resolves but is not tradeable; the index CFDs in DISCOVER_TARGETS
        # are the tradeable route. Recorded by --discover, skipped here.
        return None
    if "." in s:
        # International listing (e.g. 0QF.L). SMART routes it off the symbol
        # root; currency is unknown up front so let IBKR pick.
        return Stock(s.split(".")[0], "SMART"), "equity", "DAY"
    return Stock(s, "SMART", "USD"), "equity", "DAY"


_ERRORS: list = []


def attach_error_capture(ib):
    """Record IBKR rejection text so a failure says WHY.

    Without this every failure looks alike: whatIf just returns [] and the
    reason ("no KID for your country", "currency leverage", "off tick grid")
    is only ever delivered as an out-of-band error event.
    """
    def _on_error(reqId, code, msg, contract):
        if code not in (2104, 2106, 2107, 2158, 2119):   # connection chatter
            _ERRORS.append((code, " ".join(str(msg).split())[:150]))

    ib.errorEvent += _on_error


def probe_whatif(ib, contract, qty, price, tif, action="BUY"):
    """One non-transmitting whatIf. Returns (OrderState | None, note)."""
    from ib_async import LimitOrder, OrderState

    if qty <= 0:
        return None, "qty<=0"
    o = LimitOrder(action, qty, round(price, 5))
    o.tif = tif
    o.outsideRth = True
    _ERRORS.clear()
    try:
        st = ib.whatIfOrder(contract, o)
    except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
        return None, f"whatif_error:{str(exc)[:60]}"
    if not isinstance(st, OrderState):
        # whatIf collapses to [] on any warning (off-market limit, unsupported
        # product, out-of-band notional, no trading permission).
        if _ERRORS:
            code, msg = _ERRORS[-1]
            return None, f"{code}:{msg[:90]}"
        return None, "no_order_state"
    return st, ""


def infer_commission_model(c1, n1, c2, n2):
    """Split two (commission, notional) probes into (floor, marginal rate %).

    IBKR charges `max(floor, rate * notional)`. The small probe is normally
    floor-bound and the large one rate-bound, so the pair separates the two.
    Returns (None, None) for whichever side the data cannot support -- a
    guessed zero here would read as "this instrument is free".
    """
    if c1 is not None and c2 is not None and n1 and n2 and n2 > n1:
        if abs(c2 - c1) < 1e-6:
            # Both probes hit the same number: both are floor-bound, so the
            # marginal rate is below what this size range can resolve.
            return c1, 0.0
        rate = (c2 - c1) / (n2 - n1)
        return max(0.0, c1 - rate * n1), 100.0 * rate
    if c2 is not None and n2:
        return None, 100.0 * c2 / n2
    if c1 is not None and n1:
        return c1, None
    return None, None


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (f != f or abs(f) > 1e100) else f


def sweep_one(ib, contract, inst_class, tif, yf_symbol, price, price_date,
              in_universe, pace):
    """Resolve one contract and probe it. Returns a row dict."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    base = dict(
        yf_symbol=yf_symbol, ib_symbol=contract.symbol, sec_type=contract.secType,
        exchange=contract.exchange or "", currency=contract.currency or "",
        instrument_class=inst_class, broker="IBKR",
        included_in_universe=1 if in_universe else 0,
        ref_price=price, ref_price_date=price_date, measured_at=now,
        con_id=None, local_symbol=None, primary_exchange=None, long_name=None,
        min_tick=None, min_size=None, size_increment=None,
        small_qty=None, small_notional=None, small_commission=None,
        large_qty=None, large_notional=None, large_commission=None,
        commission_currency=None, commission_min=None, commission_rate_pct=None,
        init_margin_pct=None, maint_margin_pct=None, short_init_margin_pct=None,
        trading_hours=None, time_zone=None, status="error", note="",
    )

    try:
        details = ib.reqContractDetails(contract)
    except Exception as exc:  # noqa: BLE001
        base["status"], base["note"] = "error", f"details:{str(exc)[:70]}"
        return base
    ib.sleep(pace)
    if not details:
        base["status"] = "no_contract"
        return base

    d = details[0]
    con = d.contract
    base.update(
        ib_symbol=con.symbol, con_id=con.conId, local_symbol=con.localSymbol,
        exchange=con.exchange or "", currency=con.currency or "",
        primary_exchange=con.primaryExchange or "", long_name=d.longName or "",
        min_tick=_num(d.minTick), min_size=_num(d.minSize),
        size_increment=_num(d.sizeIncrement),
        trading_hours=(d.tradingHours or "")[:200], time_zone=d.timeZoneId or "",
    )
    # ETFs are worth separating from common stock for cost purposes.
    if con.secType == "STK" and (d.stockType or "").upper() == "ETF":
        base["instrument_class"] = "etf"

    # Prefer IBKR's own (delayed) price over the local mirror. The mirror is
    # patchy -- AAPL's last daily bar was 8 days old and 5.3% off, which both
    # risks an off-market rejection and skews every margin percentage, since
    # those are computed against qty*price while IBKR uses the real price.
    live = snapshot_price(ib, con, pace)
    if live and live > 0:
        price, base["ref_price"], base["ref_price_date"] = live, live, "ibkr_delayed"

    if price is None or price <= 0:
        base["status"] = "no_price"
        return base

    # A limit price off the contract's tick grid is rejected outright (error
    # 110), and whatIf then collapses to [] with no commission. BTC ticks at
    # 0.25, EUR.USD at 0.00005 -- a raw close conforms to neither.
    tick = base["min_tick"]
    if tick and tick > 0:
        price = round(round(price / tick) * tick, 10)
        if price <= 0:
            base["status"] = "no_price"
            return base

    mult = float(con.multiplier or 1)
    unit = price * mult

    # Always whole units. `sizeIncrement` says 0.0001 for AAPL and 0.01 for
    # EUR.USD, but the API rejects any fractional quantity outright (10243 /
    # 10318) -- that is a desktop-only capability. Round up to the contract's
    # own increment when it is coarser than 1 (futures spreads, some ETFs).
    raw_step = base["size_increment"] or 1.0
    step = int(raw_step) if raw_step >= 1 else 1

    def size_for(target_usd):
        if unit <= 0:
            return 0.0
        n = int(round((target_usd / unit) / step)) * step
        return float(max(step, n))

    small_qty, large_qty = size_for(PROBE_SMALL_USD), size_for(PROBE_LARGE_USD)
    notes = []

    # Spot FX: BUY is always rejected ("FX trade would expose account to
    # currency leverage") because the account holds only EUR, so buying a pair
    # shorts the quote currency. SELL reduces the EUR balance and is fine.
    # Every other class probes BUY, with SELL kept for the short-margin leg.
    long_action = "SELL" if con.secType == "CASH" else "BUY"
    short_action = "BUY" if long_action == "SELL" else "SELL"

    st_small, n1 = probe_whatif(ib, con, small_qty, price, tif, action=long_action)
    ib.sleep(pace)
    if n1:
        notes.append(f"small:{n1}")
    st_large, n2 = probe_whatif(ib, con, large_qty, price, tif, action=long_action)
    ib.sleep(pace)
    if n2:
        notes.append(f"large:{n2}")

    for st, qty, key in ((st_small, small_qty, "small"), (st_large, large_qty, "large")):
        if st is None:
            continue
        base[f"{key}_qty"] = qty
        base[f"{key}_notional"] = qty * unit
        base[f"{key}_commission"] = _num(st.commission)
        if st.commissionCurrency:
            base["commission_currency"] = st.commissionCurrency

    # Margin from whichever probe returned one; margin is linear in size so
    # either works as a percentage.
    for st, qty in ((st_large, large_qty), (st_small, small_qty)):
        if st is None:
            continue
        notional = qty * unit
        im, mm = _num(st.initMarginChange), _num(st.maintMarginChange)
        if im is not None and notional > 0:
            base["init_margin_pct"] = 100.0 * im / notional
        if mm is not None and notional > 0:
            base["maint_margin_pct"] = 100.0 * mm / notional
        break

    # Short-side initial margin: stocks are asymmetric (AAPL 28.49% long vs
    # 30.71% short), index/metal/FX CFDs are not.
    st_short, _ = probe_whatif(ib, con, large_qty, price, tif, action=short_action)
    ib.sleep(pace)
    if st_short is not None:
        im = _num(st_short.initMarginChange)
        notional = large_qty * unit
        if im is not None and notional > 0:
            base["short_init_margin_pct"] = 100.0 * im / notional

    floor, rate_pct = infer_commission_model(
        base["small_commission"], base["small_notional"],
        base["large_commission"], base["large_notional"],
    )
    base["commission_min"], base["commission_rate_pct"] = floor, rate_pct
    c1, c2 = base["small_commission"], base["large_commission"]

    if c1 is not None or c2 is not None:
        base["status"] = "ok"
    elif con.secType == "CRYPTO":
        # Not a failure to retry: the Paxos segment carries no margin, so
        # whatIf (a margin calculator) has nothing to return for any crypto at
        # any size. Contract metadata above is still valid and worth keeping.
        base["status"] = "no_whatif_crypto"
    else:
        base["status"] = "no_commission"
    base["note"] = ";".join(notes)[:200]
    return base


# How informative a row is. A retry that learns less than what is already
# stored must not overwrite it: when IBKR's sec-def farm dropped mid-run,
# reqContractDetails started timing out and rewrote fully-populated rows
# (contract metadata plus the exact rejection reason) as bare 'error' rows,
# destroying good data on a transient outage.
_STATUS_RANK = {
    "ok": 4,
    "no_commission": 3,      # contract resolved, rejection reason captured
    "no_whatif_crypto": 3,   # ditto, and terminal
    "no_price": 2,           # contract resolved, price unavailable
    "no_contract": 1,        # definitive: IBKR does not list it
    "error": 0,              # transient or unknown
}


def upsert(conn, row, force=False):
    """Write a row unless it would replace a strictly more informative one."""
    if not force:
        prev = conn.execute(
            "SELECT status FROM ibkr_instruments WHERE ib_symbol=? AND sec_type=? "
            "AND exchange=? AND currency=?",
            (row["ib_symbol"], row["sec_type"], row["exchange"], row["currency"]),
        ).fetchone()
        if prev and _STATUS_RANK.get(row["status"], 0) < _STATUS_RANK.get(prev[0], 0):
            return False
    cols = list(row)
    conn.execute(
        f"INSERT OR REPLACE INTO ibkr_instruments ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        [row[c] for c in cols],
    )
    conn.commit()
    return True


def already_done(conn) -> set:
    return {
        (a, b, c, d)
        for a, b, c, d in conn.execute(
            # no_whatif_crypto and no_contract are terminal, not transient --
            # retrying them just burns API calls for the same answer.
            "SELECT ib_symbol, sec_type, exchange, currency FROM ibkr_instruments "
            "WHERE status IN ('ok','no_contract','no_whatif_crypto')"
        )
    }


def _hand_curated_universe() -> dict:
    """The CANDIDATE_UNIVERSE literal as written in kairos_pipeline.py, before
    the runtime merge of the scraped international-equity CSVs."""
    import ast
    import re

    src = open(os.path.join(REPO_ROOT, "strategy", "kairos_pipeline.py")).read()
    m = re.search(r"^CANDIDATE_UNIVERSE = (\{.*?^\})", src, re.S | re.M)
    if not m:
        raise RuntimeError("could not locate the CANDIDATE_UNIVERSE literal")
    return ast.literal_eval(m.group(1))


def run_universe(ib, conn, prices, args):
    import kairos_pipeline as kp

    if args.symbols:
        universe = {"adhoc": [s.strip() for s in args.symbols.split(",") if s.strip()]}
        pairs = [(c, s) for c, syms in universe.items() for s in syms]
        _sweep_pairs(ib, conn, prices, args, pairs, set())
        return

    universe = kp.CANDIDATE_UNIVERSE
    if args.hand_curated:
        # The ~38k scraped international equities dwarf the 153 curated names.
        # Set-subtracting the scraped list would wrongly drop curated symbols
        # that also appear in it (AAPL is in both), so read the literal back
        # out of the source instead -- that is exactly the curated list.
        universe = _hand_curated_universe()
    pairs = [(cls, s) for cls, syms in universe.items() for s in syms]
    if args.limit:
        pairs = pairs[: args.limit]

    done = already_done(conn)
    print(f"[universe] {len(pairs)} symbols, {len(done)} already recorded", flush=True)
    _sweep_pairs(ib, conn, prices, args, pairs, done)


def _sweep_pairs(ib, conn, prices, args, pairs, done):
    for i, (_cls, sym) in enumerate(pairs, 1):
        mapped = yf_to_ib(sym)
        if mapped is None:
            print(f"  [{i}/{len(pairs)}] {sym:16s} -- no tradeable IB route, skipped",
                  flush=True)
            continue
        contract, inst_class, tif = mapped
        key = (contract.symbol, contract.secType, contract.exchange or "",
               contract.currency or "")
        if key in done:
            continue
        price, pdate = prices.get(sym, (None, None))
        row = sweep_one(ib, contract, inst_class, tif, sym, price, pdate,
                        True, args.pace)
        written = upsert(conn, row, force=args.force)
        c = row["small_commission"]
        print(f"  [{i}/{len(pairs)}] {sym:16s} {row['sec_type']:6s} {row['status']:16s} "
              f"{'' if written else '(kept better prior row) '}"
              f"comm={'-' if c is None else format(c, '.4f')} "
              f"margin={'-' if row['init_margin_pct'] is None else format(row['init_margin_pct'], '.2f') + '%'}"
              f"{'  ' + row['note'][:60] if row['note'] else ''}",
              flush=True)


def run_discover(ib, conn, prices, args):
    from ib_async import Contract, Crypto, Forex

    done = already_done(conn)
    targets = []
    for sec, sym, exch, cur, cls in DISCOVER_TARGETS:
        targets.append((Contract(secType=sec, symbol=sym, exchange=exch,
                                 currency=cur), cls, "DAY", None))
    for pair in DISCOVER_FX:
        targets.append((Forex(pair), "fx", "DAY", f"{pair}=X"))
    for sym in DISCOVER_CRYPTO:
        targets.append((Crypto(sym, "PAXOS", "USD"), "crypto", "IOC", f"{sym}-USD"))
    if args.limit:
        targets = targets[: args.limit]

    print(f"[discover] {len(targets)} targets", flush=True)
    for i, (contract, cls, tif, yf_sym) in enumerate(targets, 1):
        key = (contract.symbol, contract.secType, contract.exchange or "",
               contract.currency or "")
        if key in done:
            continue
        # No pre-snapshot here: sweep_one already takes a delayed snapshot off
        # the qualified contract and falls back to this mirror price, so
        # fetching one first just doubles the market-data calls.
        price, pdate = (prices.get(yf_sym, (None, None)) if yf_sym else (None, None))
        row = sweep_one(ib, contract, cls, tif, yf_sym, price, pdate,
                        False, args.pace)
        upsert(conn, row, force=args.force)
        print(f"  [{i}/{len(targets)}] {contract.symbol:10s} {row['sec_type']:6s} "
              f"{row['status']:14s} margin="
              f"{'-' if row['init_margin_pct'] is None else format(row['init_margin_pct'], '.2f') + '%'}",
              flush=True)


def snapshot_price(ib, con, pace):
    """Delayed snapshot price for an already-qualified contract, or None if it
    has no market-data entitlement."""
    try:
        ticker = ib.reqMktData(con, "", True, False)
        ib.sleep(max(4.0, pace))
        ib.cancelMktData(con)
        for cand in (ticker.last, ticker.close, ticker.marketPrice(), ticker.bid):
            v = _num(cand)
            # IBKR returns -1.0 as "no data available" on delayed/halted
            # feeds. Treating that as a price silently overwrites the good
            # mirror fallback with a negative number.
            if v and v > 0:
                return v
    except Exception:  # noqa: BLE001
        return None
    return None


def report(conn):
    print("=== ibkr_instruments ===")
    for line in conn.execute(
        "SELECT instrument_class, status, COUNT(*), "
        "ROUND(AVG(commission_min), 4), ROUND(AVG(init_margin_pct), 2) "
        "FROM ibkr_instruments GROUP BY instrument_class, status "
        "ORDER BY instrument_class, status"
    ):
        cls, status, n, cmin, margin = line
        print(f"  {cls:12s} {status:14s} n={n:<6} avg_min_comm={cmin} avg_margin={margin}")
    total = conn.execute("SELECT COUNT(*) FROM ibkr_instruments").fetchone()[0]
    inuni = conn.execute(
        "SELECT COUNT(*) FROM ibkr_instruments WHERE included_in_universe=1"
    ).fetchone()[0]
    print(f"  TOTAL {total} rows ({inuni} in universe, {total - inuni} discovered)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", action="store_true", help="sweep CANDIDATE_UNIVERSE")
    ap.add_argument("--discover", action="store_true", help="probe extra IBKR segments")
    ap.add_argument("--report", action="store_true", help="summarise the table and exit")
    ap.add_argument("--hand-curated", action="store_true",
                    help="universe mode: skip the ~38k scraped international equities")
    ap.add_argument("--symbols", default="",
                    help="comma-separated yfinance tickers, for ad-hoc probing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pace", type=float, default=1.5,
                    help="seconds between IBKR requests (default 1.5, be civil)")
    ap.add_argument("--force", action="store_true",
                    help="allow a retry to overwrite a more informative existing row "
                         "(normally refused, so a transient outage cannot destroy data)")
    ap.add_argument("--request-timeout", type=float, default=45.0,
                    help="seconds to wait for any single IBKR request (default 45; "
                         "0 waits forever, which is ib_async's default and hangs "
                         "the sweep if a data farm drops)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=42)
    args = ap.parse_args()

    conn = connect_db()
    if args.report:
        report(conn)
        return
    if not (args.universe or args.discover):
        ap.error("pick --universe, --discover or --report")

    from ib_async import IB

    prices = load_prices()
    print(f"loaded {len(prices):,} local prices", flush=True)
    ib = IB()
    # ib_async waits forever by default (RequestTimeout = 0). When IBKR's
    # backend data farms drop -- which they did mid-sweep, alongside a
    # "competing live session" error -- an in-flight request simply never
    # resolves and the whole sweep parks silently. A bounded wait turns that
    # into one recorded failure the next pass retries.
    ib.RequestTimeout = args.request_timeout
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=25)
    attach_error_capture(ib)
    ib.reqMarketDataType(3)   # delayed is fine for sizing; no subscriptions here
    print(f"connected to {args.host}:{args.port} (server {ib.client.serverVersion()})",
          flush=True)
    try:
        if args.universe:
            run_universe(ib, conn, prices, args)
        if args.discover:
            run_discover(ib, conn, prices, args)
    finally:
        ib.disconnect()
        report(conn)


if __name__ == "__main__":
    main()
