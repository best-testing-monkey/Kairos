import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import numpy as np
import pandas as pd
import pytest

from kairos_backtest import KairosDistribution, Direction
from kairos_meta import MultiFactorRankStrategy, _factor_zscores, PCAResidualReversalStrategy, _pca_residuals


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


# ============================================================================
# PCA Residual Reversal Strategy
# ============================================================================

class TestPCAResiduals:
    """Test the _pca_residuals helper directly."""

    def test_pca_residuals_orthogonal_to_factor_scores(self):
        """Residuals should be orthogonal to factor scores (dot product ~ 0)."""
        np.random.seed(42)
        n_assets, n_dates = 4, 100
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (n_dates, n_assets)),
            columns=["A", "B", "C", "D"]
        )
        residuals, factor_scores, loadings = _pca_residuals(returns, k=1)

        # Demean returns for this check
        returns_demeaned = returns.values - returns.values.mean(axis=0)

        # Residuals @ factor_scores (transposed) should have small dot products
        # Each residual (per date, per asset) dotted with factor scores (per date)
        for asset in residuals.columns:
            res_series = residuals[asset].values  # (n_dates,)
            dot_prod = np.abs(np.dot(res_series, factor_scores[:, 0]))
            # Should be very close to zero (orthogonal)
            assert dot_prod < 1e-10, f"Asset {asset} dot product {dot_prod} not close to zero"

    def test_pca_residuals_reconstruction_accurate(self):
        """Residuals + reconstruction should equal original demeaned returns."""
        np.random.seed(43)
        n_assets, n_dates = 3, 50
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (n_dates, n_assets)),
            columns=["X", "Y", "Z"]
        )
        residuals, factor_scores, loadings = _pca_residuals(returns, k=1)

        returns_demeaned = returns.values - returns.values.mean(axis=0)
        reconstruction = factor_scores @ loadings.T

        # residuals + reconstruction = original demeaned
        reconstructed_sum = residuals.values + reconstruction
        np.testing.assert_allclose(reconstructed_sum, returns_demeaned, atol=1e-12)

    def test_pca_residuals_k2_more_variance_explained(self):
        """With k=2, more variance should be explained (smaller residuals)."""
        np.random.seed(44)
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (100, 4)),
            columns=["A", "B", "C", "D"]
        )
        res1, _, _ = _pca_residuals(returns, k=1)
        res2, _, _ = _pca_residuals(returns, k=2)

        # Sum of squared residuals should be smaller with k=2
        sse1 = float(np.sum(res1.values ** 2))
        sse2 = float(np.sum(res2.values ** 2))
        assert sse2 < sse1, f"k=2 ({sse2}) should have smaller SSE than k=1 ({sse1})"


class TestPCAResidualReversalStrategy:
    """Test the PCAResidualReversalStrategy."""

    def _ctx(self, symbol, panel=None):
        return {"symbol": symbol,
                "returns_window": panel if panel is not None else make_panel()}

    def test_planted_shock_fires_short_when_bearish(self):
        """
        Plant an idiosyncratic shock: one asset has anomalous returns over last 5 days
        while others move together on factor. Kronos bearish -> SHORT candidate.
        """
        np.random.seed(50)
        n = 70
        rng = np.random.default_rng(50)

        # Create a panel with strong factor (first 3 assets)
        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),  # Follows factor + noise
            "BBB": factor_returns + rng.normal(0, 0.001, n),  # Follows factor + noise
            "CCC": factor_returns + rng.normal(0, 0.001, n),  # Follows factor + noise
            "DDD": factor_returns + rng.normal(0, 0.001, n),  # Follows factor + noise
        })

        # Plant idiosyncratic shock to one asset in last 5 days
        # This creates residual returns that are orthogonal to the factor
        shock = rng.normal(0.05, 0.01, 5)  # Strong positive idiosyncratic shock
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bearish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.metadata["residual_z"] > 0.5

    def test_planted_shock_returns_none_when_bullish(self):
        """Same shock but Kronos bullish -> None (gating disagreement)."""
        np.random.seed(51)
        n = 70
        rng = np.random.default_rng(51)

        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),
            "BBB": factor_returns + rng.normal(0, 0.001, n),
            "CCC": factor_returns + rng.normal(0, 0.001, n),
            "DDD": factor_returns + rng.normal(0, 0.001, n),
        })

        shock = rng.normal(0.05, 0.01, 5)
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bullish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is None  # Disagreement: bullish + SHORT residual

    def test_small_z_returns_none(self):
        """When |z| < z_entry, return None."""
        np.random.seed(52)
        panel = make_panel(n=70, seed=52)  # No shock, z should be small

        strat = PCAResidualReversalStrategy(window=60, z_entry=10.0, k=1)  # Very high threshold
        dist = make_bullish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is None  # Unshocked asset should have small z

    def test_missing_context_returns_none(self):
        strat = PCAResidualReversalStrategy()
        dist = make_bullish_dist()
        assert strat.generate_signal(dist, 100.0, make_history(), {}) is None
        assert strat.generate_signal(dist, 100.0, make_history(), None) is None

    def test_short_window_returns_none(self):
        """Window < 60 days returns None."""
        short_panel = make_panel(n=40, seed=60)
        strat = PCAResidualReversalStrategy(window=60)
        dist = make_bullish_dist()
        assert strat.generate_signal(dist, 100.0, make_history(),
                                      self._ctx("AAA", short_panel)) is None

    def test_fewer_than_3_assets_returns_none(self):
        """Need >= 3 assets for PCA; < 3 returns None."""
        np.random.seed(61)
        # Only 2 assets
        small_panel = pd.DataFrame(
            np.random.normal(0, 0.01, (70, 2)),
            columns=["X", "Y"]
        )
        strat = PCAResidualReversalStrategy(window=60)
        dist = make_bullish_dist()
        assert strat.generate_signal(dist, 100.0, make_history(),
                                      self._ctx("X", small_panel)) is None

    def test_symbol_not_in_returns_returns_none(self):
        panel = make_panel()
        strat = PCAResidualReversalStrategy()
        dist = make_bullish_dist()
        sig = strat.generate_signal(dist, 100.0, make_history(),
                                     self._ctx("ZZZZ", panel))
        assert sig is None

    def test_negative_shock_fires_long_when_bullish(self):
        """
        Plant negative idiosyncratic shock and Kronos bullish -> LONG candidate.
        """
        np.random.seed(62)
        n = 70
        rng = np.random.default_rng(62)

        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),
            "BBB": factor_returns + rng.normal(0, 0.001, n),
            "CCC": factor_returns + rng.normal(0, 0.001, n),
            "DDD": factor_returns + rng.normal(0, 0.001, n),
        })

        # Negative idiosyncratic shock
        shock = rng.normal(-0.05, 0.01, 5)
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bullish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.metadata["residual_z"] < -0.5

    def test_confidence_scales_with_z(self):
        """Confidence should scale with |z|, capped at 1.0."""
        np.random.seed(63)
        n = 70
        rng = np.random.default_rng(63)

        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),
            "BBB": factor_returns + rng.normal(0, 0.001, n),
            "CCC": factor_returns + rng.normal(0, 0.001, n),
            "DDD": factor_returns + rng.normal(0, 0.001, n),
        })

        # Big idiosyncratic shock
        shock = rng.normal(0.08, 0.01, 5)
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bearish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is not None
        # confidence = min(|z| / 4, 1.0)
        expected_conf = min(abs(sig.metadata["residual_z"]) / 4.0, 1.0)
        assert abs(sig.confidence - expected_conf) < 1e-6

    def test_size_respects_kelly_cap(self):
        """Size = min(kelly * 0.5, 1.0)."""
        np.random.seed(64)
        n = 70
        rng = np.random.default_rng(64)

        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),
            "BBB": factor_returns + rng.normal(0, 0.001, n),
            "CCC": factor_returns + rng.normal(0, 0.001, n),
            "DDD": factor_returns + rng.normal(0, 0.001, n),
        })

        shock = rng.normal(0.05, 0.01, 5)
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bearish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is not None
        assert 0.0 <= sig.size <= 1.0

    def test_bracket_reversed_for_short(self):
        """For SHORT, bracket should be stop > target."""
        np.random.seed(65)
        n = 70
        rng = np.random.default_rng(65)

        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),
            "BBB": factor_returns + rng.normal(0, 0.001, n),
            "CCC": factor_returns + rng.normal(0, 0.001, n),
            "DDD": factor_returns + rng.normal(0, 0.001, n),
        })

        shock = rng.normal(0.05, 0.01, 5)
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bearish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.stop > sig.target  # Reversed bracket for SHORT

    def test_bracket_normal_for_long(self):
        """For LONG, bracket should be stop < target."""
        np.random.seed(66)
        n = 70
        rng = np.random.default_rng(66)

        factor_returns = rng.normal(0, 0.005, n)
        panel = pd.DataFrame({
            "AAA": factor_returns + rng.normal(0, 0.001, n),
            "BBB": factor_returns + rng.normal(0, 0.001, n),
            "CCC": factor_returns + rng.normal(0, 0.001, n),
            "DDD": factor_returns + rng.normal(0, 0.001, n),
        })

        shock = rng.normal(-0.05, 0.01, 5)
        panel.loc[panel.index[-5:], "AAA"] += shock

        strat = PCAResidualReversalStrategy(window=60, z_entry=0.5, k=1)
        dist = make_bullish_dist(price=100.0)

        sig = strat.generate_signal(dist, 100.0, make_history(), self._ctx("AAA", panel))
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.stop < sig.target  # Normal bracket for LONG
