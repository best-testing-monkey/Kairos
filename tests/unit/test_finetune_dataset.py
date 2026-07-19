"""Tests for the multi-symbol _OHLCVDataset / ConcatDataset combination added
to kairos/cli/finetune.py (Part 1 of the automated-finetuning plan).

CPU-only, tiny synthetic frames - no GPU, no network, no real Kronos model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from kairos.cli.finetune import _OHLCVDataset, _set_epoch_seed

LOOKBACK = 10
PREDICT = 3
WINDOW = LOOKBACK + PREDICT + 1  # 14


def _make_ohlcv_frame(n_bars: int, base_price: float, seed: int = 0) -> pd.DataFrame:
    """Build a tiny synthetic OHLCV frame with a DatetimeIndex. `base_price`
    keeps two symbols' numeric ranges far apart so any accidental cross-symbol
    contamination would be obvious if it were to occur."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="D")
    close = base_price + rng.normal(0, 0.5, size=n_bars).cumsum()
    open_ = close + rng.normal(0, 0.1, size=n_bars)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, size=n_bars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, size=n_bars))
    volume = rng.uniform(1000, 2000, size=n_bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestOHLCVDatasetSingleSymbol:
    """Sanity checks on the (unchanged) single-symbol dataset behavior."""

    def test_window_count_matches_expected(self):
        df = _make_ohlcv_frame(60, base_price=100.0)
        ds = _OHLCVDataset(df, split="train", lookback_window=LOOKBACK, predict_window=PREDICT)
        # train split = first 70% of 60 rows = 42 rows; samples = 42 - 14 + 1 = 29
        assert ds.n_samples == 42 - WINDOW + 1

    def test_getitem_shapes(self):
        df = _make_ohlcv_frame(60, base_price=100.0)
        ds = _OHLCVDataset(df, split="train", lookback_window=LOOKBACK, predict_window=PREDICT)
        x, x_stamp = ds[0]
        assert isinstance(x, torch.Tensor) and isinstance(x_stamp, torch.Tensor)
        assert x.shape == (WINDOW, len(_OHLCVDataset.FEATURES))
        assert x_stamp.shape == (WINDOW, len(_OHLCVDataset.TIME_FEATURES))

    def test_amount_column_auto_derived(self):
        # _OHLCVDataset must synthesize 'amount' = close * volume when absent.
        df = _make_ohlcv_frame(60, base_price=100.0)
        assert "amount" not in df.columns
        ds = _OHLCVDataset(df, split="train", lookback_window=LOOKBACK, predict_window=PREDICT)
        assert "amount" in ds.data.columns


class TestOHLCVDatasetMultiSymbolConcat:
    """Verifies the multi-symbol combination pattern used by finetune.py's
    main(): one _OHLCVDataset built per symbol, combined per-split via
    ConcatDataset - so no sliding window ever spans a symbol boundary."""

    def _build_pair(self, split):
        # Sized so both train and val splits have a positive sample count for
        # both symbols (val split is the smaller ~15% slice).
        df_a = _make_ohlcv_frame(200, base_price=100.0, seed=1)
        df_b = _make_ohlcv_frame(150, base_price=100_000.0, seed=2)
        ds_a = _OHLCVDataset(df_a, split=split, lookback_window=LOOKBACK, predict_window=PREDICT)
        ds_b = _OHLCVDataset(df_b, split=split, lookback_window=LOOKBACK, predict_window=PREDICT)
        return ds_a, ds_b

    def test_train_concat_length_is_sum_of_per_symbol_samples(self):
        ds_a, ds_b = self._build_pair("train")
        assert ds_a.n_samples > 0 and ds_b.n_samples > 0
        concat = ConcatDataset([ds_a, ds_b])
        assert len(concat) == ds_a.n_samples + ds_b.n_samples

    def test_val_concat_length_is_sum_of_per_symbol_samples(self):
        ds_a, ds_b = self._build_pair("val")
        assert ds_a.n_samples > 0 and ds_b.n_samples > 0
        concat = ConcatDataset([ds_a, ds_b])
        assert len(concat) == ds_a.n_samples + ds_b.n_samples

    def test_no_window_spans_symbol_boundary(self):
        """Every index in the ConcatDataset must resolve to exactly the same
        sample its originating single-symbol dataset would produce directly -
        i.e. ConcatDataset never blends rows from the two underlying frames
        into a single window."""
        ds_a, ds_b = self._build_pair("train")
        concat = ConcatDataset([ds_a, ds_b])

        # Indices [0, len(ds_a)) must come from ds_a...
        for i in (0, ds_a.n_samples // 2, ds_a.n_samples - 1):
            x_concat, stamp_concat = concat[i]
            x_direct, stamp_direct = ds_a[i]
            assert torch.equal(x_concat, x_direct)
            assert torch.equal(stamp_concat, stamp_direct)

        # ...and indices [len(ds_a), len(concat)) must come from ds_b.
        for i in (0, ds_b.n_samples // 2, ds_b.n_samples - 1):
            concat_idx = ds_a.n_samples + i
            x_concat, stamp_concat = concat[concat_idx]
            x_direct, stamp_direct = ds_b[i]
            assert torch.equal(x_concat, x_direct)
            assert torch.equal(stamp_concat, stamp_direct)

    def test_set_epoch_seed_helper_covers_every_underlying_dataset(self):
        """The _set_epoch_seed helper (used by _train's per-epoch reseed) must
        reach every underlying _OHLCVDataset inside a ConcatDataset - the
        outer ConcatDataset has no set_epoch_seed of its own."""
        ds_a, ds_b = self._build_pair("train")
        concat = ConcatDataset([ds_a, ds_b])
        _set_epoch_seed(concat, 5)
        assert ds_a._epoch == 5
        assert ds_b._epoch == 5

    def test_set_epoch_seed_helper_single_dataset_unchanged(self):
        """Single-symbol invocations (a plain _OHLCVDataset, not wrapped in a
        ConcatDataset) must keep working identically."""
        df = _make_ohlcv_frame(60, base_price=100.0)
        ds = _OHLCVDataset(df, split="train", lookback_window=LOOKBACK, predict_window=PREDICT)
        _set_epoch_seed(ds, 3)
        assert ds._epoch == 3
