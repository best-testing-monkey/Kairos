import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import Direction, Signal, Strategy
from kairos_ml import MetaLabelStrategy


# ============================================================================
# Helpers
# ============================================================================

class FakeDist:
    """Duck-typed stand-in for KairosDistribution with controllable features."""

    def __init__(self, entropy=1.0, kurt=0.0, skew=0.0, cdf=0.5):
        self._entropy = entropy
        self._cdf = cdf
        self.stats = {"close": {"kurt": kurt, "skew": skew}}

    def entropy(self, col="close", bins=20):
        return self._entropy

    def cdf(self, price, col="close"):
        return self._cdf


class StubStrategy(Strategy):
    """Base strategy that always emits a fixed LONG signal."""
    name = "stub"

    def __init__(self, emit=True):
        self.emit = emit

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        if not self.emit:
            return None
        return Signal(
            direction=Direction.LONG, size=0.5, entry=current_price,
            stop=current_price * 0.95, target=current_price * 1.10,
            strategy_name=self.name, confidence=0.8, expected_value=1.0,
            metadata={"origin": "stub"},
        )


def make_history(n=50, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [price] * n, "high": [price * 1.01] * n,
        "low": [price * 0.99] * n, "close": [price] * n, "volume": [1e6] * n,
    }, index=idx)


def auc_score(y_true, y_score):
    """AUC via the rank-sum (Mann-Whitney U) statistic, numpy only."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ranks for ties
    for v in np.unique(y_score):
        mask = y_score == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def drive_labeled_samples(strat, n, rng, history):
    """Feed n labeled samples: win iff entropy < 2.0."""
    entropies = rng.uniform(0.0, 3.0, size=n)
    for e in entropies:
        dist = FakeDist(
            entropy=e,
            kurt=rng.normal(0, 1),
            skew=rng.normal(0, 0.5),
            cdf=rng.uniform(0, 1),
        )
        strat.generate_signal(dist, 100.0, history, {"regime_id": rng.integers(0, 3)})
        strat.label_last(1.0 if e < 2.0 else 0.0)
    return entropies


# ============================================================================
# Tests
# ============================================================================

class TestMetaLabelStrategy:

    def test_meta_label_none_base_passthrough(self):
        """Base strategy returning None passes through as None."""
        strat = MetaLabelStrategy(base_strategy=StubStrategy(emit=False))
        sig = strat.generate_signal(FakeDist(), 100.0, make_history(), {})
        assert sig is None

    def test_meta_label_warm_up_passthrough(self):
        """During warm-up the base signal passes through with identical fields."""
        base = StubStrategy()
        strat = MetaLabelStrategy(base_strategy=base, warmup=60)
        history = make_history()
        expected = base.generate_signal(FakeDist(), 100.0, history, {})

        rng = np.random.default_rng(0)
        for i in range(30):  # well below warmup=60
            dist = FakeDist(entropy=rng.uniform(0, 3))
            sig = strat.generate_signal(dist, 100.0, history, {})
            assert sig is not None
            assert sig.direction == expected.direction
            assert sig.size == expected.size
            assert sig.entry == expected.entry
            assert sig.stop == expected.stop
            assert sig.target == expected.target
            assert sig.confidence == expected.confidence
            assert sig.expected_value == expected.expected_value
            strat.label_last(1.0)

    def test_meta_label_entropy_classifier_auc(self):
        """Wins iff entropy < 2.0: model reaches AUC > 0.9 on held-out data."""
        rng = np.random.default_rng(42)
        history = make_history()
        strat = MetaLabelStrategy(base_strategy=StubStrategy(), warmup=60)

        drive_labeled_samples(strat, 200, rng, history)
        assert strat.classifier.fitted

        # Held-out synthetic set
        n_test = 200
        entropies = rng.uniform(0.0, 3.0, size=n_test)
        X_test, y_test = [], []
        for e in entropies:
            dist = FakeDist(entropy=e, kurt=rng.normal(0, 1),
                            skew=rng.normal(0, 0.5), cdf=rng.uniform(0, 1))
            feats = strat._extract_features(dist, 100.0, history,
                                            {"regime_id": rng.integers(0, 3)})
            X_test.append(feats)
            y_test.append(1.0 if e < 2.0 else 0.0)

        p = strat.classifier.predict_proba(np.array(X_test))
        assert auc_score(y_test, p) > 0.9

    def test_meta_label_veto_below_threshold(self):
        """After training, high-entropy signals are vetoed, low-entropy pass."""
        rng = np.random.default_rng(7)
        history = make_history()
        strat = MetaLabelStrategy(base_strategy=StubStrategy(), p_min=0.55, warmup=60)
        drive_labeled_samples(strat, 200, rng, history)

        # High entropy (losing regime) -> vetoed
        sig_high = strat.generate_signal(FakeDist(entropy=2.9), 100.0, history, {})
        assert sig_high is None

        # Low entropy (winning regime) -> passes, size scaled by p_win
        sig_low = strat.generate_signal(FakeDist(entropy=0.5), 100.0, history, {})
        assert sig_low is not None
        assert sig_low.direction == Direction.LONG
        p_win = sig_low.metadata["p_win"]
        assert p_win >= strat.p_min
        assert sig_low.size == pytest.approx(0.5 * p_win)

    def test_meta_label_reset_clears_history(self):
        """reset() clears labeled history, fit state, and pending signal."""
        rng = np.random.default_rng(3)
        history = make_history()
        strat = MetaLabelStrategy(base_strategy=StubStrategy(), warmup=10)
        drive_labeled_samples(strat, 50, rng, history)
        assert len(strat.labeled_pairs) == 50
        assert strat.fit_count > 0

        strat.reset()
        assert strat.labeled_pairs == []
        assert strat.fit_count == 0
        assert strat.pending_signal is None
        assert strat.pending_features is None
        assert not strat.classifier.fitted
        assert len(strat.trailing_outcomes) == 0

        # Back to warm-up pass-through behavior
        sig = strat.generate_signal(FakeDist(entropy=2.9), 100.0, history, {})
        assert sig is not None
        assert sig.size == 0.5

    def test_meta_label_refit_cadence(self):
        """Model refits every 20 new labels, never per call."""
        rng = np.random.default_rng(11)
        history = make_history()
        strat = MetaLabelStrategy(base_strategy=StubStrategy(), warmup=60)

        drive_labeled_samples(strat, 19, rng, history)
        assert strat.fit_count == 0  # not yet at cadence

        drive_labeled_samples(strat, 1, rng, history)
        assert strat.fit_count == 1  # fit at 20 labels

        drive_labeled_samples(strat, 19, rng, history)
        assert strat.fit_count == 1  # no per-label refitting

        drive_labeled_samples(strat, 1, rng, history)
        assert strat.fit_count == 2  # fit at 40 labels
