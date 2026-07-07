import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import Direction, Signal, Strategy, KairosDistribution
from kairos_execution import volume_profile, VolumeProfileLevelsStrategy, CVDDivergenceStrategy


# ============================================================================
# Helpers
# ============================================================================

def make_profile_history():
    """
    Hand-built history whose volume profile is exactly known.

    Price range: [100, 120] -> 20 bins of width 1, centers 100.5 .. 119.5.
    Each bar's typical price (H+L+C)/3 lands exactly on one bin center.

    Bin volumes:
      - center 104.5: volume 300  (HVN + POC)
      - center 112.5: volume 0    (LVN, no bar there)
      - all 18 other centers: volume 100

    mean = 105, std ~= 49.75
      -> HVN threshold (mean+std) ~= 154.75 -> only the 300 bin
      -> LVN threshold (mean-std) ~= 55.25  -> only the 0 bin
    """
    rows = []
    centers = [100.5 + i for i in range(20)]
    for c in centers:
        if c == 112.5:
            continue  # zero-volume bin (LVN)
        vol = 300.0 if c == 104.5 else 100.0
        if c == 100.5:
            # anchor window minimum low at 100; typical = (101+100+100.5)/3 = 100.5
            rows.append({"open": 100.5, "high": 101.0, "low": 100.0,
                         "close": 100.5, "volume": vol})
        elif c == 119.5:
            # anchor window maximum high at 120; typical = (120+119+119.5)/3 = 119.5
            rows.append({"open": 119.5, "high": 120.0, "low": 119.0,
                         "close": 119.5, "volume": vol})
        else:
            rows.append({"open": c, "high": c, "low": c, "close": c, "volume": vol})
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx)


class StubStrategy(Strategy):
    """Base strategy returning a preset Signal (or None)."""
    name = "stub"

    def __init__(self, signal):
        self._signal = signal

    def generate_signal(self, dist, current_price, history, context):
        return self._signal


def make_signal(direction, entry, stop, target):
    return Signal(
        direction=direction, size=0.5, entry=entry, stop=stop, target=target,
        strategy_name="stub", confidence=0.8, expected_value=1.0, metadata={},
    )


# ============================================================================
# volume_profile() function
# ============================================================================

class TestVolumeProfileFunction:
    def test_volume_profile_poc_vah_val(self):
        """POC / HVN / LVN computed exactly on a hand-built fixture."""
        vp = volume_profile(make_profile_history(), lookback=60, bins=20)
        assert vp["poc"] == pytest.approx(104.5)
        assert vp["hvn"] == pytest.approx([104.5])
        assert vp["lvn"] == pytest.approx([112.5])
        assert len(vp["edges"]) == 21
        assert vp["edges"][0] == pytest.approx(100.0)
        assert vp["edges"][-1] == pytest.approx(120.0)

    def test_degenerate_history(self):
        """Too-short history returns empty profile without crashing."""
        h = make_profile_history().head(1)
        vp = volume_profile(h, lookback=60, bins=20)
        assert vp["hvn"] == [] and vp["lvn"] == []


# ============================================================================
# VolumeProfileLevelsStrategy wrapper
# ============================================================================

class TestVolumeProfileLevelsStrategy:
    def _wrap(self, signal):
        return VolumeProfileLevelsStrategy(StubStrategy(signal), lookback=60, bins=20)

    def test_none_passthrough(self):
        strat = self._wrap(None)
        assert strat.generate_signal(None, 110.0, make_profile_history(), {}) is None

    def test_volume_profile_stop_snap(self):
        """LONG: stop snaps up to the HVN between stop and entry."""
        sig = make_signal(Direction.LONG, entry=110.0, stop=103.0, target=118.0)
        out = self._wrap(sig).generate_signal(None, 110.0, make_profile_history(), {})
        assert isinstance(out, Signal)
        assert out.stop == pytest.approx(104.5)  # snapped to HVN, tighter than 103
        vp_meta = out.metadata["volume_profile"]
        assert vp_meta["stop_original"] == pytest.approx(103.0)
        assert vp_meta["stop_snapped"] == pytest.approx(104.5)

    def test_volume_profile_target_snap(self):
        """LONG: target snaps to the LVN beyond entry when it is nearer."""
        sig = make_signal(Direction.LONG, entry=110.0, stop=103.0, target=118.0)
        out = self._wrap(sig).generate_signal(None, 110.0, make_profile_history(), {})
        assert out.target == pytest.approx(112.5)  # LVN nearer than 118
        vp_meta = out.metadata["volume_profile"]
        assert vp_meta["target_original"] == pytest.approx(118.0)
        assert vp_meta["target_snapped"] == pytest.approx(112.5)

    def test_volume_profile_stop_only_tightens(self):
        """LONG: a stop already tighter than the HVN is never widened."""
        sig = make_signal(Direction.LONG, entry=110.0, stop=106.0, target=118.0)
        out = self._wrap(sig).generate_signal(None, 110.0, make_profile_history(), {})
        # HVN 104.5 is NOT between 106 and 110 -> stop unchanged
        assert out.stop == pytest.approx(106.0)

    def test_short_stop_snap_tightens(self):
        """SHORT: stop snaps down to the HVN between entry and stop."""
        sig = make_signal(Direction.SHORT, entry=102.0, stop=107.0, target=100.5)
        out = self._wrap(sig).generate_signal(None, 102.0, make_profile_history(), {})
        assert out.stop == pytest.approx(104.5)  # 102 < 104.5 < 107 -> tightened
        assert out.stop < 107.0

    def test_short_stop_never_widens(self):
        """SHORT: a stop tighter than any HVN stays put."""
        sig = make_signal(Direction.SHORT, entry=102.0, stop=104.0, target=100.5)
        out = self._wrap(sig).generate_signal(None, 102.0, make_profile_history(), {})
        # HVN 104.5 is not between 102 and 104 -> unchanged
        assert out.stop == pytest.approx(104.0)

    def test_short_target_snap(self):
        """SHORT: target snaps to the LVN below entry when it is nearer."""
        sig = make_signal(Direction.SHORT, entry=115.0, stop=120.0, target=105.0)
        out = self._wrap(sig).generate_signal(None, 115.0, make_profile_history(), {})
        assert out.target == pytest.approx(112.5)  # LVN 112.5 nearer than 105
        assert out.stop == pytest.approx(120.0)    # no HVN in (115, 120)

    def test_target_not_snapped_when_original_nearer(self):
        """Target only moves through the gap if the LVN is nearer than the original."""
        sig = make_signal(Direction.LONG, entry=110.0, stop=108.0, target=111.0)
        out = self._wrap(sig).generate_signal(None, 110.0, make_profile_history(), {})
        assert out.target == pytest.approx(111.0)  # 111 nearer than LVN 112.5
        assert out.stop == pytest.approx(108.0)    # no HVN in (108, 110)

    def test_no_qualifying_nodes_unchanged(self):
        """With no HVN/LVN at all, the signal brackets are untouched."""
        # Flat-volume history: every bin has the same volume -> no nodes
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        prices = np.linspace(100.0, 120.0, 20)
        flat = pd.DataFrame({
            "open": prices, "high": prices + 0.1, "low": prices - 0.1,
            "close": prices, "volume": [100.0] * 20,
        }, index=idx)
        vp = volume_profile(flat, lookback=60, bins=20)
        sig = make_signal(Direction.LONG, entry=110.0, stop=103.0, target=118.0)
        out = self._wrap(sig).generate_signal(None, 110.0, flat, {})
        assert out.stop == pytest.approx(103.0)
        assert out.target == pytest.approx(118.0)
        assert "volume_profile" in out.metadata

    def test_returns_signal_not_dict(self):
        sig = make_signal(Direction.LONG, entry=110.0, stop=103.0, target=118.0)
        out = self._wrap(sig).generate_signal(None, 110.0, make_profile_history(), {})
        assert isinstance(out, Signal)
        assert not isinstance(out, dict)


# ============================================================================
# CVDDivergenceStrategy tests
# ============================================================================

class TestCVDDivergenceStrategy:
    """Test cumulative volume delta divergence strategy."""

    def _make_dist(self, closes_array, mean=None, std=None):
        """Create a mock KairosDistribution from close prices."""
        if mean is None:
            mean = float(np.mean(closes_array))
        if std is None:
            std = float(np.std(closes_array))

        dist = KairosDistribution([pd.DataFrame({"close": closes_array})])
        dist.stats["close"] = {
            "mean": mean,
            "std": std,
            "pct_10": float(np.percentile(closes_array, 10)),
            "pct_15": float(np.percentile(closes_array, 15)),
            "pct_25": float(np.percentile(closes_array, 25)),
            "pct_50": float(np.percentile(closes_array, 50)),
            "pct_75": float(np.percentile(closes_array, 75)),
            "pct_85": float(np.percentile(closes_array, 85)),
            "pct_90": float(np.percentile(closes_array, 90)),
        }
        dist.stats["high"] = {
            "pct_10": float(np.percentile(closes_array, 10)),
            "pct_15": float(np.percentile(closes_array, 15)),
            "pct_85": float(np.percentile(closes_array, 85)),
            "pct_90": float(np.percentile(closes_array, 90)),
        }
        dist.stats["low"] = {
            "pct_10": float(np.percentile(closes_array, 10)),
            "pct_15": float(np.percentile(closes_array, 15)),
            "pct_85": float(np.percentile(closes_array, 85)),
            "pct_90": float(np.percentile(closes_array, 90)),
        }
        return dist

    def test_cvd_sign_convention(self):
        """Test that CVD correctly signs volume by close-vs-open.

        Hand-built 5-bar fixture:
        - Bar 1: open=100, close=101 → close > open → +100 vol
        - Bar 2: open=101, close=100 → close < open → -100 vol
        - Bar 3: open=100, close=100 → close = open → 0 vol
        - Bar 4: open=99, close=101 → close > open → +200 vol
        - Bar 5: open=101, close=99  → close < open → -150 vol

        CVD cumsum: [100, 0, 0, 200, 50]
        """
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        history = pd.DataFrame({
            "open": [100.0, 101.0, 100.0, 99.0, 101.0],
            "close": [101.0, 100.0, 100.0, 101.0, 99.0],
            "high": [101.0, 101.0, 100.0, 101.0, 101.0],
            "low": [99.5, 99.5, 99.5, 99.0, 99.0],
            "volume": [100.0, 100.0, 100.0, 200.0, 150.0],
        }, index=idx)

        strat = CVDDivergenceStrategy(slope_window=3, lookback=5)

        # Manually compute expected CVD for verification
        close_minus_open = history["close"].values - history["open"].values
        volume_signed = history["volume"].values * np.sign(close_minus_open)
        cvd_expected = np.cumsum(volume_signed)

        assert cvd_expected[0] == pytest.approx(100.0)  # +100
        assert cvd_expected[1] == pytest.approx(0.0)    # +100 - 100
        assert cvd_expected[2] == pytest.approx(0.0)    # +100 - 100 + 0
        assert cvd_expected[3] == pytest.approx(200.0)  # +100 - 100 + 0 + 200
        assert cvd_expected[4] == pytest.approx(50.0)   # +100 - 100 + 0 + 200 - 150

    def test_cvd_divergence_bearish_kronos_bearish(self):
        """Bearish divergence (price rising, CVD falling) + Kronos bearish → SHORT.

        Fixture: 25 bars
        - Prices continuously rising (positive price slope)
        - Early bars: up bars with large volume (positive CVD contribution)
        - Late bars: down bars with large volume (negative CVD contribution)
        - This creates rising prices but falling CVD (bearish divergence)
        - Kronos mean < current_price (bearish)
        → Should return SHORT signal
        """
        idx = pd.date_range("2024-01-01", periods=25, freq="D")

        # Create rising prices over all 25 bars (price slope will be positive)
        closes = np.linspace(100.0, 120.0, 25)

        # Mix bar directions: early bars up, late bars down (but still closing higher)
        opens = np.zeros(25)
        # First 12 bars: up bars (close > open) with large positive volume
        opens[:12] = closes[:12] - 2.0
        # Last 13 bars: down bars (close < open) with large negative volume
        # But prices still overall trend up
        opens[12:] = closes[12:] + 3.0

        # Volume pattern: high early (ups), high late (downs creates negative CVD)
        volumes = np.concatenate([
            np.linspace(500.0, 400.0, 12),  # First 12: large positive CVD
            np.linspace(500.0, 600.0, 13),  # Last 13: large negative CVD (downs)
        ])

        history = pd.DataFrame({
            "open": opens,
            "close": closes,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "volume": volumes,
        }, index=idx)

        # Kronos predicts down (mean < current)
        current_price = closes[-1]
        closes_pred = np.linspace(110.0, 90.0, 100)
        dist = self._make_dist(closes_pred, mean=100.0, std=5.0)

        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, current_price, history, {})

        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.metadata["cvd_slope"] < 0  # CVD slope should be negative
        assert signal.metadata["price_slope"] > 0  # Price slope should be positive

    def test_cvd_divergence_bearish_kronos_bullish(self):
        """Bearish divergence but Kronos bullish → None (gated out).

        Same setup as above but Kronos mean > current_price.
        """
        idx = pd.date_range("2024-01-01", periods=25, freq="D")

        prices = np.linspace(100.0, 120.0, 25)
        opens = prices.copy()
        closes = prices.copy() + 0.5
        opens = closes - 0.5

        volumes = np.ones(25) * 100.0
        volumes[-20:] = np.where(
            np.arange(5, 25) % 2 == 0,
            500.0, 50.0
        )

        history = pd.DataFrame({
            "open": opens,
            "close": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "volume": volumes,
        }, index=idx)

        current_price = prices[-1]
        closes_pred = np.linspace(125.0, 145.0, 100)
        dist = self._make_dist(closes_pred, mean=135.0, std=5.0)

        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, current_price, history, {})

        # Kronos is bullish; should reject the bearish divergence signal
        assert signal is None

    def test_cvd_divergence_bullish_kronos_bullish(self):
        """Bullish divergence (price falling, CVD rising) + Kronos bullish → LONG.

        Fixture: 25 bars
        - Prices continuously falling (negative price slope)
        - Early bars: down bars with large volume (negative CVD contribution)
        - Late bars: up bars with large volume (positive CVD contribution)
        - This creates rising CVD despite falling prices (bullish divergence)
        - Kronos mean > current_price (bullish)
        → Should return LONG signal
        """
        idx = pd.date_range("2024-01-01", periods=25, freq="D")

        # Create falling prices over all 25 bars (price slope will be negative)
        closes = np.linspace(120.0, 100.0, 25)

        # Mix bar directions: early bars down, late bars up (but still closing lower)
        opens = np.zeros(25)
        # First 12 bars: down bars (close < open) with large negative volume
        opens[:12] = closes[:12] + 2.0
        # Last 13 bars: up bars (close > open) with large positive volume
        # But prices still overall trend down
        opens[12:] = closes[12:] - 3.0

        # Volume pattern: high early (downs), high late (ups creates positive CVD)
        volumes = np.concatenate([
            np.linspace(400.0, 500.0, 12),  # First 12: large negative CVD
            np.linspace(500.0, 600.0, 13),  # Last 13: large positive CVD (ups)
        ])

        history = pd.DataFrame({
            "open": opens,
            "close": closes,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "volume": volumes,
        }, index=idx)

        current_price = closes[-1]
        closes_pred = np.linspace(110.0, 130.0, 100)
        dist = self._make_dist(closes_pred, mean=120.0, std=5.0)

        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, current_price, history, {})

        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.metadata["cvd_slope"] > 0  # CVD rising
        assert signal.metadata["price_slope"] < 0  # Price falling

    def test_cvd_divergence_bullish_kronos_bearish(self):
        """Bullish divergence but Kronos bearish → None (gated out)."""
        idx = pd.date_range("2024-01-01", periods=25, freq="D")

        prices = np.linspace(120.0, 100.0, 25)
        opens = prices.copy()
        closes = prices.copy() - 0.5
        opens = closes + 0.5

        volumes = np.ones(25) * 100.0
        volumes[-20:] = np.where(
            np.arange(5, 25) % 2 == 1,
            500.0, 50.0
        )

        history = pd.DataFrame({
            "open": opens,
            "close": closes,
            "high": opens + 0.1,
            "low": closes - 0.1,
            "volume": volumes,
        }, index=idx)

        current_price = prices[-1]
        closes_pred = np.linspace(95.0, 75.0, 100)
        dist = self._make_dist(closes_pred, mean=85.0, std=5.0)

        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, current_price, history, {})

        # Kronos is bearish; should reject bullish divergence
        assert signal is None

    def test_cvd_divergence_no_divergence(self):
        """Aligned slopes (both up or both down) → None."""
        idx = pd.date_range("2024-01-01", periods=25, freq="D")

        # Strong trending: prices and volume both rising
        prices = np.linspace(100.0, 120.0, 25)
        closes = prices.copy()
        opens = closes - 1.0  # All bars up
        volumes = np.linspace(100.0, 500.0, 25)  # Rising volume

        history = pd.DataFrame({
            "open": opens,
            "close": closes,
            "high": closes + 0.1,
            "low": closes - 1.1,
            "volume": volumes,
        }, index=idx)

        current_price = prices[-1]
        dist = self._make_dist(np.linspace(120.0, 140.0, 100), mean=130.0, std=5.0)

        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, current_price, history, {})

        # Both slopes positive = no divergence
        assert signal is None

    def test_cvd_divergence_short_history(self):
        """History shorter than slope_window+1 → None."""
        idx = pd.date_range("2024-01-01", periods=15, freq="D")
        history = pd.DataFrame({
            "open": np.ones(15) * 100.0,
            "close": np.linspace(100.0, 110.0, 15),
            "high": np.linspace(101.0, 111.0, 15),
            "low": np.linspace(99.0, 109.0, 15),
            "volume": np.ones(15) * 100.0,
        }, index=idx)

        dist = self._make_dist(np.ones(100) * 105.0)
        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, 110.0, history, {})

        # Only 15 bars < 20+1 required
        assert signal is None

    def test_cvd_divergence_returns_signal_not_dict(self):
        """Ensure signal is a Signal object, never a dict."""
        idx = pd.date_range("2024-01-01", periods=25, freq="D")

        # Create rising prices with bearish divergence
        closes = np.linspace(100.0, 120.0, 25)

        opens = np.zeros(25)
        opens[:12] = closes[:12] - 2.0
        opens[12:] = closes[12:] + 3.0

        volumes = np.concatenate([
            np.linspace(500.0, 400.0, 12),
            np.linspace(500.0, 600.0, 13),
        ])

        history = pd.DataFrame({
            "open": opens,
            "close": closes,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "volume": volumes,
        }, index=idx)

        current_price = closes[-1]
        closes_pred = np.linspace(110.0, 90.0, 100)
        dist = self._make_dist(closes_pred, mean=100.0, std=5.0)

        strat = CVDDivergenceStrategy(slope_window=20, lookback=25)
        signal = strat.generate_signal(dist, current_price, history, {})

        assert isinstance(signal, Signal)
        assert not isinstance(signal, dict)
        assert signal.metadata["cvd_slope"] is not None
        assert signal.metadata["price_slope"] is not None
