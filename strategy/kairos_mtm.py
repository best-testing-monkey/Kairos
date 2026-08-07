"""kairos_mtm.py — Pure mark-to-market snapshot math.

Operates on plain dataclasses and price bars. Imports no phantom, GPU, or
network libraries.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from kairos.errors import KairosError
from kairos_margin import classify_symbol, MarginConfig


@dataclass(frozen=True)
class OpenPositionView:
    """Read-only view of an open position for MTM calculations.

    Attributes:
        ticker: Symbol or contract identifier.
        direction: ``"long"`` or ``"short"``.
        entry_price: Fill price at which the position was opened.
        quantity: Number of units (always positive; direction encodes sign).
        entry_costs: Commission + spread + slippage + fx conversion costs paid
            on entry.
    """

    ticker: str
    direction: str
    entry_price: float
    quantity: float
    entry_costs: float


@dataclass(frozen=True)
class DailySnapshot:
    """Portfolio-level mark-to-market state for a single day.

    Attributes:
        date: Calendar date of the snapshot (taken from the supplied bars).
        cash: Settled cash available in the account.
        unrealized_pnl: Sum of direction-aware unrealized PnL across positions.
        equity: ``cash + unrealized_pnl``.
        gross_notional: Sum of ``entry_price * quantity`` for all open positions.
        initial_margin_used: Aggregate initial margin requirement.
        maintenance_margin_used: Aggregate maintenance margin requirement.
        free_margin: ``equity - initial_margin_used``.
        margin_utilization: ``initial_margin_used / equity`` when equity > 0,
            otherwise ``0.0``.
        financing_accrued_day: Financing cost accrued during this day; kept at
            ``0.0`` for E2-S03.
        liquidations: Number of positions liquidated on this day; kept at ``0``
            for E2-S03.
    """

    date: datetime.date
    cash: float
    unrealized_pnl: float
    equity: float
    gross_notional: float
    initial_margin_used: float
    maintenance_margin_used: float
    free_margin: float
    margin_utilization: float
    financing_accrued_day: float
    liquidations: int


Bar = dict[str, Any]


def unrealized_pnl(pos: OpenPositionView, close_price: float) -> float:
    """Return direction-aware unrealized PnL for a single position.

    long: ``(close_price - entry_price) * quantity``
    short: ``(entry_price - close_price) * quantity``

    Args:
        pos: Open position view.
        close_price: Mark price for the ticker on the snapshot date.

    Returns:
        Unrealized profit/loss in account currency units.

    Raises:
        KairosError: If ``direction`` is not ``"long"`` or ``"short"``.
    """
    direction = pos.direction.lower()
    if direction == "long":
        return (close_price - pos.entry_price) * pos.quantity
    if direction == "short":
        return (pos.entry_price - close_price) * pos.quantity
    raise KairosError(f"Unknown direction {pos.direction!r} for {pos.ticker!r}")


def _bar_close(bar: Bar) -> float:
    """Extract the closing price from a bar dictionary."""
    try:
        return float(bar["close"])
    except (KeyError, TypeError) as exc:
        raise KairosError(f"Bar is missing a numeric 'close' value: {exc}") from exc


def _bar_date(bar: Bar) -> datetime.date:
    """Extract the date from a bar dictionary."""
    value = bar.get("date")
    if value is None:
        raise KairosError("Bar is missing a 'date' value")
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(value)


def compute_daily_snapshot(
    positions: list[OpenPositionView],
    bars_by_ticker: dict[str, Bar],
    cash: float,
    cfg: MarginConfig,
) -> DailySnapshot:
    """Compute a portfolio MTM snapshot from open positions and closing bars.

    Args:
        positions: Open positions to mark.
        bars_by_ticker: Mapping from ticker to a bar containing at least
            ``"date"`` and ``"close"`` keys.
        cash: Current settled cash balance.
        cfg: Loaded margin configuration.

    Returns:
        ``DailySnapshot`` with all fields populated.

    Raises:
        KairosError: If a required bar is missing, a bar lacks price/date data,
            or no positions/bars are supplied to infer the snapshot date.
    """
    if not positions:
        if not bars_by_ticker:
            raise KairosError(
                "Cannot compute snapshot date without positions or bars"
            )
        empty_date = _bar_date(next(iter(bars_by_ticker.values())))
        return DailySnapshot(
            date=empty_date,
            cash=cash,
            unrealized_pnl=0.0,
            equity=cash,
            gross_notional=0.0,
            initial_margin_used=0.0,
            maintenance_margin_used=0.0,
            free_margin=cash,
            margin_utilization=0.0,
            financing_accrued_day=0.0,
            liquidations=0,
        )

    snap_date: datetime.date | None = None
    gross_notional = 0.0
    initial_margin_used = 0.0
    maintenance_margin_used = 0.0
    total_unrealized = 0.0

    for pos in positions:
        bar = bars_by_ticker.get(pos.ticker)
        if bar is None:
            raise KairosError(f"No bar available for {pos.ticker!r}")

        if snap_date is None:
            snap_date = _bar_date(bar)

        close_price = _bar_close(bar)
        margin_class = classify_symbol(pos.ticker, cfg)

        notional = pos.entry_price * pos.quantity
        gross_notional += notional
        initial_margin_used += notional * margin_class.initial_margin_pct / 100.0
        maintenance_margin_used += notional * margin_class.maintenance_margin_pct / 100.0
        total_unrealized += unrealized_pnl(pos, close_price)

    if snap_date is None:
        raise KairosError("No positions to determine snapshot date")

    equity = cash + total_unrealized
    free_margin = equity - initial_margin_used
    margin_utilization = initial_margin_used / equity if equity > 0 else 0.0

    return DailySnapshot(
        date=snap_date,
        cash=cash,
        unrealized_pnl=total_unrealized,
        equity=equity,
        gross_notional=gross_notional,
        initial_margin_used=initial_margin_used,
        maintenance_margin_used=maintenance_margin_used,
        free_margin=free_margin,
        margin_utilization=margin_utilization,
        financing_accrued_day=0.0,
        liquidations=0,
    )
