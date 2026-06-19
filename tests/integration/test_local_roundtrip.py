"""KAI-8: Integration smoke — local SQLite round trip (no network).

Uses the seeded_db fixture from conftest.py.
"""
import numpy as np
import pandas as pd
import pytest

import kairos
import price_cache


class TestLocalRoundtrip:
    def test_forecast_window_shape(self, seeded_db):
        db_path, ticker, start, end = seeded_db
        price_cache.DB_PATH = db_path
        kairos.configure(remote=False)

        end_str = end.isoformat()
        lookback = 30
        pred_len = 5

        x_df, x_ts, y_ts = kairos.get_forecast_window(
            ticker, "1d", lookback, pred_len, end=end_str
        )

        assert len(x_df) == lookback
        assert len(x_ts) == lookback
        assert len(y_ts) == pred_len

    def test_forecast_window_contract(self, seeded_db):
        db_path, ticker, start, end = seeded_db
        price_cache.DB_PATH = db_path
        kairos.configure(remote=False)

        x_df, x_ts, y_ts = kairos.get_forecast_window(
            ticker, "1d", 20, 3, end=end.isoformat()
        )

        # Columns
        assert set(x_df.columns) == {"open", "high", "low", "close", "volume"}
        # dtypes
        for col in x_df.columns:
            assert x_df[col].dtype == np.float64
        # No NaNs
        assert not x_df.isna().any().any()
        # Timestamps
        assert pd.DatetimeIndex(x_ts).is_monotonic_increasing
        assert pd.DatetimeIndex(y_ts).is_monotonic_increasing
        assert y_ts.iloc[0] > x_ts.iloc[-1]

    def test_forecast_window_amount_close_volume(self, seeded_db):
        db_path, ticker, start, end = seeded_db
        price_cache.DB_PATH = db_path
        kairos.configure(remote=False)

        x_df, _, _ = kairos.get_forecast_window(
            ticker, "1d", 10, 2, end=end.isoformat(), amount="close_volume"
        )
        assert "amount" in x_df.columns
        expected = x_df["close"] * x_df["volume"]
        pd.testing.assert_series_equal(x_df["amount"], expected, check_names=False)
