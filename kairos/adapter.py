"""KAI-2: Schema adapter - price_cache frame → Kronos OHLCV contract.

Pure function; no I/O, no global state.
"""
from __future__ import annotations

import pandas as pd

from .errors import DataQualityError, InsufficientDataError

_RENAME: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
_REQUIRED: list[str] = ["open", "high", "low", "close", "volume"]


def to_kronos_frame(
    df: pd.DataFrame,
    lookback: int,
    amount: str = "omit",
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a price_cache DataFrame to the Kronos OHLCV input contract.

    Args:
        df: Raw price_cache frame (capitalized column names, DatetimeIndex).
        lookback: Exact number of bars required.
        amount: One of "omit" | "auto" | "close_volume".

    Returns:
        (x_df, x_timestamp): Kronos-shaped frame and corresponding timestamps.

    Raises:
        InsufficientDataError: Fewer than *lookback* rows.
        DataQualityError: NaN in any OHLCV cell.
    """
    df = df.tail(lookback).copy()
    if len(df) < lookback:
        raise InsufficientDataError(have=len(df), want=lookback)

    out = df.rename(columns=_RENAME)[_REQUIRED].astype("float64")

    if amount == "auto":
        if "amount" in df.columns and df["amount"].notna().all():
            out["amount"] = df["amount"].astype("float64")
        elif "vwap" in df.columns and df["vwap"].notna().all():
            out["amount"] = (df["vwap"] * df["Volume"]).astype("float64")
        else:
            out["amount"] = out["close"] * out["volume"]
    elif amount == "close_volume":
        out["amount"] = out["close"] * out["volume"]
    # "omit" → no amount column

    if out[_REQUIRED].isna().any().any():
        raise DataQualityError("NaN in OHLCV after adaptation")

    x_timestamp = pd.Series(df.index, name="x_timestamp")
    return out.reset_index(drop=True), x_timestamp
