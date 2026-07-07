import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import Direction, Signal, Strategy
from kairos_execution import volume_profile, VolumeProfileLevelsStrategy


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
