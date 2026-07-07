import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import numpy as np
import pandas as pd
import pytest

from kairos_backtest import KairosDistribution, Direction
from kairos_meta import MultiFactorRankStrategy, _factor_zscores


# ============================================================================
# Helpers
# ============================================================================

def make_dist(close_prices):
    """Build a KairosDistribution from a list of close prices."""
    prices = np.array(close_prices, dtype=float)
    frames = []
    for p in prices:
        frames.append(pd.DataFrame({
            "open": [p * 0.999], "high": [p * 1.005], "low": [p * 0.995],
            "close": [p], "volume": [1e6], "amount": [1e9]
        }))
    return KairosDistribution(frames)


def make_bullish_dist(price=100.0):
    """Kronos distribution with mean well above current price."""
    np.random.seed(1)
    return make_dist(np.random.normal(price * 1.03, price * 0.02, 100))


def make_bearish_dist(price=100.0):
    """Kronos distribution with mean well below current price."""
    np.random.seed(2)
    return make_dist(np.random.normal(price * 0.97, price * 0.02, 100))


def make_panel(n=300, seed=7):
    """
    Synthetic 4-asset returns panel.

    AAA: planted strong momentum + quality (steady positive, low vol) -> top.
    BBB, CCC: pure noise -> middle.
    DDD: steady negative drift, high vol -> bottom.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "AAA": 0.004 + rng.normal(0, 0.002, n),
        "BBB": rng.normal(0, 0.010, n),
        "CCC": rng.normal(0, 0.010, n),
        "DDD": -0.004 + rng.normal(0, 0.014, n),
    })


def make_history(n=50, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [price] * n, "high": [price * 1.01] * n,
        "low": [price * 0.99] * n, "close": [price] * n, "volume": [1e6] * n
    }, index=idx)


# ============================================================================
# Z-scoring helper
# ============================================================================

class TestFactorZScores:
    def test_multi_factor_z_scoring(self):
        """Hand-computed 3-asset, 2-factor case."""
        panel = pd.DataFrame(
            {"momentum": [1.0, 2.0, 3.0], "quality": [10.0, 10.0, 10.0]},
            index=["X", "Y", "Z"],
        )
        z = _factor_zscores(panel)
        # momentum: mean=2, population std=sqrt(2/3)
        expected = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0 / 3.0)
        np.testing.assert_allclose(z["momentum"].values, expected, atol=1e-12)
        # constant factor -> all zeros, no division blowup
        np.testing.assert_allclose(z["quality"].values, 0.0)

    def test_zscores_have_zero_mean_unit_std(self):
        panel = pd.DataFrame(
            {"f": [3.5, -1.2, 0.0, 9.9]}, index=list("ABCD"))
        z = _factor_zscores(panel)["f"].values
        assert abs(np.mean(z)) < 1e-12
        assert abs(np.std(z) - 1.0) < 1e-12


# ============================================================================
# MultiFactorRankStrategy signal logic
# ============================================================================

class TestMultiFactorRankStrategy:
    def _ctx(self, symbol, panel=None):
        return {"symbol": symbol,
                "returns_window": panel if panel is not None else make_panel()}

    def test_top_ranked_bullish_emits_long(self):
        strat = MultiFactorRankStrategy()
        sig = strat.generate_signal(
            make_bullish_dist(), 100.0, make_history(), self._ctx("AAA"))
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.strategy_name == "multi_factor_rank"
        assert sig.metadata["rank_top"] == "AAA"
        assert sig.metadata["rank_bottom"] == "DDD"
        # bracket: stop = pct_15, target = pct_85 of predicted close
        assert sig.stop < sig.target
        assert 0.0 < sig.confidence <= 1.0
        assert 0.0 <= sig.size <= 1.0

    def test_middle_ranked_returns_none(self):
        strat = MultiFactorRankStrategy()
        for sym in ("BBB", "CCC"):
            sig = strat.generate_signal(
                make_bullish_dist(), 100.0, make_history(), self._ctx(sym))
            assert sig is None

    def test_bottom_ranked_bearish_emits_short(self):
        strat = MultiFactorRankStrategy()
        sig = strat.generate_signal(
            make_bearish_dist(), 100.0, make_history(), self._ctx("DDD"))
        assert sig is not None
        assert sig.direction == Direction.SHORT
        # reversed bracket for SHORT
        assert sig.stop > sig.target

    def test_kronos_disagreement_returns_none(self):
        strat = MultiFactorRankStrategy()
        # top-ranked but Kronos bearish
        assert strat.generate_signal(
            make_bearish_dist(), 100.0, make_history(), self._ctx("AAA")) is None
        # bottom-ranked but Kronos bullish
        assert strat.generate_signal(
            make_bullish_dist(), 100.0, make_history(), self._ctx("DDD")) is None

    def test_missing_context_returns_none(self):
        strat = MultiFactorRankStrategy()
        dist = make_bullish_dist()
        assert strat.generate_signal(dist, 100.0, make_history(), {}) is None
        assert strat.generate_signal(dist, 100.0, make_history(), None) is None
        # symbol present but no panel
        assert strat.generate_signal(
            dist, 100.0, make_history(), {"symbol": "AAA"}) is None
        # panel present but symbol not in it
        assert strat.generate_signal(
            dist, 100.0, make_history(), self._ctx("ZZZ")) is None

    def test_short_window_returns_none(self):
        strat = MultiFactorRankStrategy()
        short_panel = make_panel(n=40)
        assert strat.generate_signal(
            make_bullish_dist(), 100.0, make_history(),
            self._ctx("AAA", short_panel)) is None

    def test_universe_history_fallback(self):
        """universe_history dict of close DataFrames works as fallback."""
        panel = make_panel()
        histories = {
            sym: pd.DataFrame({"close": 100.0 * np.cumprod(1 + panel[sym].values)})
            for sym in panel.columns
        }
        strat = MultiFactorRankStrategy()
        sig = strat.generate_signal(
            make_bullish_dist(), 100.0, make_history(),
            {"symbol": "AAA", "universe_history": histories})
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_multi_factor_momentum_only_degenerates(self):
        """With momentum-only weights, composite ordering = momentum ordering
        (single-factor rank behavior, like CrossAssetRankStrategy)."""
        panel = make_panel()
        strat = MultiFactorRankStrategy(
            weights={"momentum": 1.0, "low_vol": 0.0, "value": 0.0, "quality": 0.0})
        factors = strat._compute_factors(panel)
        composite = strat._composite(_factor_zscores(factors))
        assert list(composite.sort_values(ascending=False).index) == \
            list(factors["momentum"].sort_values(ascending=False).index)
        # and the top-momentum asset is the one traded
        sig = strat.generate_signal(
            make_bullish_dist(), 100.0, make_history(),
            {"symbol": factors["momentum"].idxmax(), "returns_window": panel})
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_equal_weights_factors_contribute_equally(self):
        """Composite with equal weights is the plain mean of factor z-scores."""
        panel = make_panel()
        strat = MultiFactorRankStrategy()
        z = _factor_zscores(strat._compute_factors(panel))
        composite = strat._composite(z)
        np.testing.assert_allclose(
            composite.values, z.values.mean(axis=1), atol=1e-12)
