"""Fixtures: seed a local SQLite DB with synthetic OHLCV data."""
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

_TICKER = "FIXTURE"
_START = date(2023, 1, 3)   # first NYSE trading day of 2023
_END = date(2024, 3, 29)    # last NYSE trading day of Q1 2024


def _trading_days(start: date, end: date):
    """Simple weekday-only generator (no holiday filter needed for fixtures)."""
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            yield cur
        cur += timedelta(days=1)


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory):
    """Return path to a SQLite DB pre-loaded with synthetic daily prices."""
    db_dir = tmp_path_factory.mktemp("db")
    db_path = str(db_dir / "fixture.db")

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker           TEXT NOT NULL,
            date             TEXT NOT NULL,
            interval_minutes INTEGER NOT NULL DEFAULT 1440,
            open             REAL,
            high             REAL,
            low              REAL,
            close            REAL,
            volume           INTEGER,
            dividends        REAL,
            stock_splits     REAL,
            market_cap       INTEGER,
            PRIMARY KEY (ticker, date, interval_minutes)
        );
        CREATE TABLE IF NOT EXISTS fetched_ranges (
            ticker           TEXT NOT NULL,
            fetched_from     TEXT NOT NULL,
            fetched_to       TEXT NOT NULL,
            interval_minutes INTEGER NOT NULL DEFAULT 1440,
            UNIQUE(ticker, fetched_from, fetched_to, interval_minutes)
        );
        CREATE INDEX IF NOT EXISTS idx_fetched_ticker ON fetched_ranges(ticker);
        CREATE TABLE IF NOT EXISTS no_data_tickers (
            ticker TEXT PRIMARY KEY, noted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS ticker_lookup (
            company_name TEXT PRIMARY KEY, ticker TEXT, searched_at TEXT
        );
        CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER PRIMARY KEY);
        INSERT OR IGNORE INTO _schema_version VALUES (3);
    """)

    days = list(_trading_days(_START, _END))
    price = 100.0
    rows = []
    for i, d in enumerate(days):
        price = max(1.0, price + (i % 7 - 3) * 0.5)
        rows.append((
            _TICKER, d.isoformat(), 1440,
            round(price - 0.5, 4), round(price + 1.0, 4),
            round(price - 1.0, 4), round(price, 4),
            100_000 + i * 10, 0.0, 0.0, 0,
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    # Mark the full realistic window as already fetched so price_cache never
    # tries to fill gaps from the network during tests.
    conn.execute(
        "INSERT OR IGNORE INTO fetched_ranges VALUES (?,?,?,?)",
        (_TICKER, "1990-01-01", _END.isoformat(), 1440),
    )
    conn.commit()
    conn.close()

    return db_path, _TICKER, _START, _END
