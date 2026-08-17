"""kairos_mtm.py — Pure mark-to-market snapshot math.

Operates on plain dataclasses and price bars. Imports no phantom, GPU, or
network libraries.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from kairos.errors import KairosError
from allocation import AllocationConfig
from kairos_margin import classify_symbol, MarginClass, MarginConfig


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


def position_margin_contribution(pos: OpenPositionView, cfg: MarginConfig) -> tuple[float, float, float]:
    """Return ``(notional, initial_margin, maintenance_margin)`` for one position.

    Same per-position margin math used inside ``compute_daily_snapshot``,
    exposed standalone so callers can fold a position's margin usage into a
    snapshot without also needing a mark-to-market close price -- e.g. a
    same-day round-trip position that already closed, where marking it to a
    close price would double-count P&L already reflected in cash.

    Args:
        pos: Position view (entry price/quantity/ticker only; direction and
            entry_costs are unused here).
        cfg: Loaded margin configuration.

    Returns:
        Tuple of ``(notional, initial_margin, maintenance_margin)``.
    """
    margin_class = classify_symbol(pos.ticker, cfg)
    notional = pos.entry_price * pos.quantity
    initial_margin = notional * margin_class.initial_margin_pct / 100.0
    maintenance_margin = notional * margin_class.maintenance_margin_pct / 100.0
    return notional, initial_margin, maintenance_margin


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

    Equity note: a full-notional (``initial_margin_pct >= 100``, e.g. spot
    crypto, or any position when the caller is in cash-only/``max_leverage
    <= 1.0`` mode) position has its FULL notional debited from ``cash`` at
    entry (see ``kairos_papertrade.py``'s ``_use_full_notional``/
    ``_fill_cash_delta``) -- unlike a margin-only position, where only the
    margin requirement leaves cash. ``equity`` must therefore add back such a
    position's full CURRENT market value (``close_price * quantity``), not
    just its P&L delta (``unrealized_pnl``) -- adding only the delta leaves
    the entire invested notional missing from equity, understating it by
    roughly the position's size. This previously caused ``margin_utilization``
    to read well over 100% (and could trip ``liquidation_check``'s ESMA
    close-out rule) for a perfectly healthy account holding open spot
    positions. The returned ``unrealized_pnl`` field itself is unaffected --
    it's still the pure P&L delta, for reporting.
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
    total_equity_contribution = 0.0

    for pos in positions:
        bar = bars_by_ticker.get(pos.ticker)
        if bar is None:
            raise KairosError(f"No bar available for {pos.ticker!r}")

        if snap_date is None:
            snap_date = _bar_date(bar)

        close_price = _bar_close(bar)
        notional, initial_margin, maintenance_margin = position_margin_contribution(pos, cfg)
        gross_notional += notional
        initial_margin_used += initial_margin
        maintenance_margin_used += maintenance_margin
        delta = unrealized_pnl(pos, close_price)
        total_unrealized += delta
        # See the equity note in this function's docstring: full-notional
        # positions contribute their full current value, margin-only
        # positions contribute just their P&L delta.
        if classify_symbol(pos.ticker, cfg).initial_margin_pct >= 100.0:
            total_equity_contribution += close_price * pos.quantity
        else:
            total_equity_contribution += delta

    if snap_date is None:
        raise KairosError("No positions to determine snapshot date")

    equity = cash + total_equity_contribution
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


def admission_check(
    order_notional: float,
    ticker: str,
    account: DailySnapshot,
    cfg: MarginConfig,
    alloc_config: AllocationConfig,
) -> bool:
    """Return True if a new order may be admitted without breaching margin limits.

    When ``alloc_config.max_leverage <= 1.0`` the check is a no-op and returns
    ``True`` to preserve legacy cash-only behavior.

    Args:
        order_notional: Notional value of the new order in account currency.
        ticker: Symbol or contract identifier (used for margin class lookup).
        account: Current daily MTM snapshot.
        cfg: Loaded margin configuration.
        alloc_config: Allocation configuration with leverage/cap settings.

    Returns:
        ``True`` if the order passes the post-trade margin admission test,
        otherwise ``False``.
    """
    if alloc_config.max_leverage <= 1.0:
        return True

    margin_class = classify_symbol(ticker, cfg)
    new_gross_notional = account.gross_notional + order_notional  # noqa: F841
    new_initial_margin_used = (
        account.initial_margin_used + order_notional * margin_class.initial_margin_pct / 100.0
    )
    new_equity = account.equity

    return (
        new_initial_margin_used <= new_equity * alloc_config.margin_utilization_cap
        and new_equity > 0.0
    )


def daily_financing(
    pos: OpenPositionView,
    close_price: float,
    cls: MarginClass,
    cfg: MarginConfig,
) -> float:
    """Return financing cost accrued on a single position for one day.

    Spot classes (``initial_margin_pct == 100.0``) accrue zero financing.

    Long CFD/margin: charged ``notional_close * (benchmark_annual_pct + financing_spread_pct) / 360``.

    Short CFD/margin: debited ``notional_close * short_borrow_annual_pct / 360``,
    minus a credit of ``(benchmark_annual_pct - financing_spread_pct) / 360``
    (credit only if positive). The returned value represents the net financing
    cost charged to the account (positive = cost owed, negative = benefit received).

    By convention, financing is charged on positions open at bar close; the entry
    day counts, the exit day does not. This function computes the daily accrual
    only; the caller's day loop enforces the entry/exit day convention.

    Args:
        pos: Open position view.
        close_price: Mark price for the position on the day.
        cls: Margin class for the position's ticker.
        cfg: Loaded margin configuration, including ``benchmark_annual_pct``
            and ``short_borrow_annual_pct``.

    Returns:
        Daily financing cost in account currency (positive = cost, negative = credit).

    Raises:
        KairosError: If ``pos.direction`` is not ``"long"`` or ``"short"``.
    """
    # Spot classes (no margin) incur no financing.
    if cls.initial_margin_pct >= 100.0:
        return 0.0

    notional_close = close_price * pos.quantity
    direction = pos.direction.lower()

    if direction == "long":
        # Long: charged (benchmark + spread) daily
        return notional_close * (cfg.benchmark_annual_pct + cls.financing_spread_pct) / 360.0

    if direction == "short":
        # Short: always debited borrow fee, credited financing if positive
        short_borrow_pct = cfg.short_borrow_annual_pct.get("overrides", {}).get(
            pos.ticker, cfg.short_borrow_annual_pct.get("default", 0.0)
        )
        borrow_cost = notional_close * short_borrow_pct / 360.0
        financing_credit = max(
            0.0,
            notional_close * (cfg.benchmark_annual_pct - cls.financing_spread_pct) / 360.0,
        )
        return borrow_cost - financing_credit

    raise KairosError(f"Unknown direction {pos.direction!r} for {pos.ticker!r}")


def compute_daily_financing_total(
    positions: list[OpenPositionView],
    bars_by_ticker: dict[str, Bar],
    cfg: MarginConfig,
) -> float:
    """Sum daily financing accrual across all open positions for a single day.

    Each position's financing is computed via ``daily_financing``, using the
    closing price from ``bars_by_ticker``.

    Args:
        positions: Open positions to accrue financing on.
        bars_by_ticker: Mapping from ticker to bar with ``"close"`` key.
        cfg: Loaded margin configuration.

    Returns:
        Total daily financing cost (positive = net cost to account).

    Raises:
        KairosError: If a required bar is missing, lacks price data, or
            a position's direction is invalid.
    """
    total = 0.0
    for pos in positions:
        bar = bars_by_ticker.get(pos.ticker)
        if bar is None:
            raise KairosError(f"No bar available for {pos.ticker!r}")

        close_price = _bar_close(bar)
        margin_class = classify_symbol(pos.ticker, cfg)
        total += daily_financing(pos, close_price, margin_class, cfg)

    return total


def liquidation_check(
    snapshot: DailySnapshot,
    positions: list[OpenPositionView],
    cfg: MarginConfig,
) -> tuple[list[str], float, bool]:
    """Determine which positions to liquidate under the ESMA 50% close-out rule.

    Evaluates the liquidation trigger condition: when account equity falls below
    ``closeout_fraction * initial_margin_used``, positions are liquidated in
    greedy order (largest maintenance-margin release first) until the condition
    is satisfied or no positions remain.

    Args:
        snapshot: Current daily MTM snapshot with equity and margin state.
        positions: List of open positions available for liquidation.
        cfg: Loaded margin configuration with ``closeout_fraction``.

    Returns:
        A tuple ``(tickers_liquidated, post_equity, ruined)`` where:
        - ``tickers_liquidated``: List of tickers that were force-closed.
        - ``post_equity``: Equity after liquidation (clamped to 0.0 if negative).
        - ``ruined``: ``True`` if all positions were liquidated and equity
          remains non-positive; ``False`` otherwise.

    Raises:
        KairosError: If a position's direction is invalid.
    """
    # Check trigger condition: equity < closeout_fraction * initial_margin_used
    if snapshot.equity >= cfg.closeout_fraction * snapshot.initial_margin_used:
        # Not triggered; return current state
        return ([], snapshot.equity, False)

    # Trigger condition met; liquidate positions greedily
    tickers_liquidated: list[str] = []
    remaining_positions = list(positions)  # work with a copy

    # Precompute maintenance margin release for each position for sorting
    maintenance_releases: list[tuple[int, str, float]] = []
    for idx, pos in enumerate(remaining_positions):
        notional = pos.entry_price * pos.quantity
        margin_class = classify_symbol(pos.ticker, cfg)
        mm_release = notional * margin_class.maintenance_margin_pct / 100.0
        maintenance_releases.append((idx, pos.ticker, mm_release))

    # Sort by maintenance margin release, descending (largest first)
    maintenance_releases.sort(key=lambda x: x[2], reverse=True)

    # Current state tracking for recomputation
    current_equity = snapshot.equity
    current_initial_margin_used = snapshot.initial_margin_used

    # Liquidate positions in order of largest MM release
    for idx, ticker, _mm_release in maintenance_releases:
        pos = remaining_positions[idx]
        notional = pos.entry_price * pos.quantity
        margin_class = classify_symbol(pos.ticker, cfg)

        # Remove this position's margin requirement
        current_initial_margin_used -= (
            notional * margin_class.initial_margin_pct / 100.0
        )

        # Record liquidation
        tickers_liquidated.append(ticker)

        # Check if we've restored safety (equity >= closeout_fraction * im_used)
        if current_equity >= cfg.closeout_fraction * current_initial_margin_used:
            break

    # Determine ruined status and clamp equity if needed
    post_equity = current_equity
    ruined = False

    if len(tickers_liquidated) == len(positions):
        # All positions were liquidated
        if post_equity <= 0.0:
            post_equity = 0.0
            ruined = True

    return (tickers_liquidated, post_equity, ruined)
