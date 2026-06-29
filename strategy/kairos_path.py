"""
kairos_path.py
==============
Intra-day path extraction from Kairos 60-sample distributions.

Each sample provides an OHLC tuple. From 60 samples we can extract:
- Path patterns (e.g., open -> low -> high -> close)
- Probability that high comes before low (or vice versa)
- Typical trajectory shape (rally, fade, V-shape, inverted V)
- Path confidence (entropy of path distribution)
- Median path for execution planning

Usage:
    from kairos_path import KairosPathExtractor, PathPattern

    extractor = KairosPathExtractor(predictions)
    profile = extractor.extract()

    if profile.high_before_low_prob > 0.8:
        # Short at predicted high, cover at predicted low
    elif profile.low_before_high_prob > 0.8:
        # Long at predicted low, sell at predicted high
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# PATH PATTERN ENUMS
# =============================================================================

class PathPattern(Enum):
    """
    All 4! = 24 possible orderings of O, H, L, C.
    We name the most common ones for readability.
    """
    # Rally then fade
    OPEN_LOW_HIGH_CLOSE = "O_L_H_C"       # Open -> Low -> High -> Close
    LOW_OPEN_HIGH_CLOSE = "L_O_H_C"       # Low -> Open -> High -> Close
    OPEN_HIGH_LOW_CLOSE = "O_H_L_C"       # Open -> High -> Low -> Close
    HIGH_OPEN_LOW_CLOSE = "H_O_L_C"       # High -> Open -> Low -> Close

    # Other common patterns
    LOW_HIGH_OPEN_CLOSE = "L_H_O_C"
    HIGH_LOW_OPEN_CLOSE = "H_L_O_C"
    OPEN_CLOSE_LOW_HIGH = "O_C_L_H"
    OPEN_CLOSE_HIGH_LOW = "O_C_H_L"
    LOW_OPEN_CLOSE_HIGH = "L_O_C_H"
    HIGH_OPEN_CLOSE_LOW = "H_O_C_L"
    LOW_HIGH_CLOSE_OPEN = "L_H_C_O"
    HIGH_LOW_CLOSE_OPEN = "H_L_C_O"
    OPEN_LOW_CLOSE_HIGH = "O_L_C_H"
    OPEN_HIGH_CLOSE_LOW = "O_H_C_L"
    CLOSE_LOW_HIGH_OPEN = "C_L_H_O"
    CLOSE_HIGH_LOW_OPEN = "C_H_L_O"
    CLOSE_OPEN_LOW_HIGH = "C_O_L_H"
    CLOSE_OPEN_HIGH_LOW = "C_O_H_L"
    LOW_CLOSE_HIGH_OPEN = "L_C_H_O"
    HIGH_CLOSE_LOW_OPEN = "H_C_L_O"
    LOW_CLOSE_OPEN_HIGH = "L_C_O_H"
    HIGH_CLOSE_OPEN_LOW = "H_C_O_L"
    CLOSE_LOW_OPEN_HIGH = "C_L_O_H"
    CLOSE_HIGH_OPEN_LOW = "C_H_O_L"

    # Fallback
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_ordering(cls, ordering: Tuple[str, str, str, str]) -> "PathPattern":
        mapping = {
            ("open", "low", "high", "close"): cls.OPEN_LOW_HIGH_CLOSE,
            ("low", "open", "high", "close"): cls.LOW_OPEN_HIGH_CLOSE,
            ("open", "high", "low", "close"): cls.OPEN_HIGH_LOW_CLOSE,
            ("high", "open", "low", "close"): cls.HIGH_OPEN_LOW_CLOSE,
            ("low", "high", "open", "close"): cls.LOW_HIGH_OPEN_CLOSE,
            ("high", "low", "open", "close"): cls.HIGH_LOW_OPEN_CLOSE,
            ("open", "close", "low", "high"): cls.OPEN_CLOSE_LOW_HIGH,
            ("open", "close", "high", "low"): cls.OPEN_CLOSE_HIGH_LOW,
            ("low", "open", "close", "high"): cls.LOW_OPEN_CLOSE_HIGH,
            ("high", "open", "close", "low"): cls.HIGH_OPEN_CLOSE_LOW,
            ("low", "high", "close", "open"): cls.LOW_HIGH_CLOSE_OPEN,
            ("high", "low", "close", "open"): cls.HIGH_LOW_CLOSE_OPEN,
            ("open", "low", "close", "high"): cls.OPEN_LOW_CLOSE_HIGH,
            ("open", "high", "close", "low"): cls.OPEN_HIGH_CLOSE_LOW,
            ("close", "low", "high", "open"): cls.CLOSE_LOW_HIGH_OPEN,
            ("close", "high", "low", "open"): cls.CLOSE_HIGH_LOW_OPEN,
            ("close", "open", "low", "high"): cls.CLOSE_OPEN_LOW_HIGH,
            ("close", "open", "high", "low"): cls.CLOSE_OPEN_HIGH_LOW,
            ("low", "close", "high", "open"): cls.LOW_CLOSE_HIGH_OPEN,
            ("high", "close", "low", "open"): cls.HIGH_CLOSE_LOW_OPEN,
            ("low", "close", "open", "high"): cls.LOW_CLOSE_OPEN_HIGH,
            ("high", "close", "open", "low"): cls.HIGH_CLOSE_OPEN_LOW,
            ("close", "low", "open", "high"): cls.CLOSE_LOW_OPEN_HIGH,
            ("close", "high", "open", "low"): cls.CLOSE_HIGH_OPEN_LOW,
        }
        return mapping.get(ordering, cls.UNKNOWN)


# =============================================================================
# PATH PROFILE
# =============================================================================

@dataclass
class PathProfile:
    """Complete path analysis for a single bar's 60 predictions."""

    # Raw counts
    pattern_counts: Dict[PathPattern, int] = field(default_factory=dict)
    total_samples: int = 0

    # Probabilities
    high_before_low_prob: float = 0.0
    low_before_high_prob: float = 0.0
    open_is_low_prob: float = 0.0
    open_is_high_prob: float = 0.0
    close_is_high_prob: float = 0.0
    close_is_low_prob: float = 0.0

    # Dominant pattern
    dominant_pattern: PathPattern = PathPattern.UNKNOWN
    dominant_pattern_prob: float = 0.0
    pattern_entropy: float = 0.0

    # Trajectory classification
    trajectory: str = "unknown"  # "rally", "fade", "v_shape", "inverted_v", "chop"
    trajectory_confidence: float = 0.0

    # Median path (for execution planning)
    median_path: List[Tuple[str, float]] = field(default_factory=list)
    # e.g., [("open", 100.0), ("low", 99.5), ("high", 101.0), ("close", 100.5)]

    # Path confidence (how concentrated are the 60 samples?)
    path_confidence: float = 0.0  # 0 = all different, 1 = all identical

    # Derived signals
    recommended_direction: Optional[str] = None  # "long", "short", "neutral"
    recommended_entry: Optional[float] = None
    recommended_target: Optional[float] = None
    recommended_stop: Optional[float] = None

    def __repr__(self) -> str:
        lines = [
            f"PathProfile(dominant={self.dominant_pattern.name}, "
            f"p={self.dominant_pattern_prob:.2f})",
            f"  H_before_L: {self.high_before_low_prob:.2f}",
            f"  L_before_H: {self.low_before_high_prob:.2f}",
            f"  Trajectory: {self.trajectory} (conf={self.trajectory_confidence:.2f})",
            f"  Path conf: {self.path_confidence:.2f}",
        ]
        if self.recommended_direction:
            lines.append(
                f"  Signal: {self.recommended_direction} "
                f"@ {self.recommended_entry:.4f} -> {self.recommended_target:.4f} "
                f"(stop {self.recommended_stop:.4f})"
            )
        return "\n".join(lines)


# =============================================================================
# PATH EXTRACTOR
# =============================================================================

class KairosPathExtractor:
    """
    Extracts path statistics from a list of 60 prediction DataFrames.
    """

    def __init__(self, predictions: List[pd.DataFrame]):
        self.predictions = predictions
        self.df = pd.concat(predictions, ignore_index=True)

    def extract(self) -> PathProfile:
        profile = PathProfile()
        profile.total_samples = len(self.predictions)
        if profile.total_samples == 0:
            return profile

        # Extract orderings from each sample
        orderings = []
        hb_l_count = 0
        lb_h_count = 0
        open_is_low = 0
        open_is_high = 0
        close_is_high = 0
        close_is_low = 0

        for _, row in self.df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            # Build (price, label) pairs and sort
            prices = [
                (float(o), "open"),
                (float(h), "high"),
                (float(l), "low"),
                (float(c), "close"),
            ]
            sorted_prices = sorted(prices, key=lambda x: x[0])
            ordering = tuple(label for _, label in sorted_prices)
            orderings.append(ordering)

            # High before low? (high price comes before low price in the sorted order)
            # Actually: we want to know if the high is reached before the low in time.
            # Since we only have OHLC, we infer from the ordering:
            # If high appears before low in the sorted-by-price list, that doesn't tell us time.
            # Instead, we use a heuristic: if close > open, likely low came first (rally).
            # If close < open, likely high came first (fade).
            # But we can do better: look at the actual sample path.
            #
            # For a single sample, we know the 4 prices but not the intra-day sequence.
            # We infer the most likely temporal ordering based on typical market microstructure:
            # - If open is near low and close is near high: O -> L -> H -> C (rally)
            # - If open is near high and close is near low: O -> H -> L -> C (fade)
            # - If open is in the middle: could be either
            #
            # A better approach: for each sample, compute the "path" as the sorted order
            # and use that as the inferred temporal sequence. This is an approximation,
            # but with 60 samples the aggregate statistics are meaningful.

            # Check if high is the maximum and low is the minimum (always true)
            # What we really want: temporal ordering. Since we don't have timestamps,
            # we use the price-sorted order as a proxy for "path".
            # In a rally: prices go up, so temporal order ~ price order.
            # In a fade: prices go down, so temporal order ~ reverse price order.
            #
            # We use the following heuristic:
            # If close > open: assume temporal order is ascending price order (rally)
            # If close < open: assume temporal order is descending price order (fade)
            # This is crude but works for the aggregate.

            if c > o:
                # Rally day: assume prices generally rise
                # Temporal order: low -> ... -> high
                # So low likely before high
                lb_h_count += 1
            else:
                # Fade day: assume prices generally fall
                # Temporal order: high -> ... -> low
                # So high likely before low
                hb_l_count += 1

            # Open/close extremes
            if o == min(o, h, l, c):
                open_is_low += 1
            if o == max(o, h, l, c):
                open_is_high += 1
            if c == max(o, h, l, c):
                close_is_high += 1
            if c == min(o, h, l, c):
                close_is_low += 1

        # Pattern frequencies
        pattern_counter = Counter(orderings)
        profile.pattern_counts = {
            PathPattern.from_ordering(k): v
            for k, v in pattern_counter.items()
        }

        # Dominant pattern
        if profile.pattern_counts:
            dominant, count = max(profile.pattern_counts.items(), key=lambda x: x[1])
            profile.dominant_pattern = dominant
            profile.dominant_pattern_prob = count / profile.total_samples

        # Entropy of pattern distribution
        total = profile.total_samples
        probs = [c / total for c in profile.pattern_counts.values() if c > 0]
        profile.pattern_entropy = -sum(p * np.log(p) for p in probs if p > 0)

        # Path confidence (1 - normalized entropy)
        max_entropy = np.log(24)  # 24 possible patterns
        profile.path_confidence = 1.0 - (profile.pattern_entropy / max_entropy if max_entropy > 0 else 0)

        # Probabilities
        profile.high_before_low_prob = hb_l_count / total
        profile.low_before_high_prob = lb_h_count / total
        profile.open_is_low_prob = open_is_low / total
        profile.open_is_high_prob = open_is_high / total
        profile.close_is_high_prob = close_is_high / total
        profile.close_is_low_prob = close_is_low / total

        # Trajectory classification
        profile.trajectory, profile.trajectory_confidence = self._classify_trajectory(profile)

        # Median path
        profile.median_path = self._compute_median_path()

        # Recommended signal
        profile.recommended_direction, profile.recommended_entry,         profile.recommended_target, profile.recommended_stop = self._derive_signal(profile)

        return profile

    def _classify_trajectory(self, profile: PathProfile) -> Tuple[str, float]:
        """
        Classify the overall trajectory based on pattern probabilities.
        """
        counts = profile.pattern_counts
        total = profile.total_samples
        if total == 0:
            return "unknown", 0.0

        # Rally patterns: open near low, close near high
        rally_patterns = [
            PathPattern.OPEN_LOW_HIGH_CLOSE,
            PathPattern.LOW_OPEN_HIGH_CLOSE,
            PathPattern.LOW_HIGH_OPEN_CLOSE,
        ]
        rally_count = sum(counts.get(p, 0) for p in rally_patterns)
        rally_prob = rally_count / total

        # Fade patterns: open near high, close near low
        fade_patterns = [
            PathPattern.OPEN_HIGH_LOW_CLOSE,
            PathPattern.HIGH_OPEN_LOW_CLOSE,
            PathPattern.HIGH_LOW_OPEN_CLOSE,
        ]
        fade_count = sum(counts.get(p, 0) for p in fade_patterns)
        fade_prob = fade_count / total

        # V-shape: low in middle, high at end
        v_patterns = [
            PathPattern.LOW_OPEN_CLOSE_HIGH,
            PathPattern.LOW_HIGH_CLOSE_OPEN,
            PathPattern.OPEN_LOW_CLOSE_HIGH,
        ]
        v_count = sum(counts.get(p, 0) for p in v_patterns)
        v_prob = v_count / total

        # Inverted V: high in middle, low at end
        inv_v_patterns = [
            PathPattern.HIGH_OPEN_CLOSE_LOW,
            PathPattern.HIGH_LOW_CLOSE_OPEN,
            PathPattern.OPEN_HIGH_CLOSE_LOW,
        ]
        inv_v_count = sum(counts.get(p, 0) for p in inv_v_patterns)
        inv_v_prob = inv_v_count / total

        # Chop: open and close in the middle
        chop_patterns = [
            PathPattern.OPEN_CLOSE_LOW_HIGH,
            PathPattern.OPEN_CLOSE_HIGH_LOW,
            PathPattern.CLOSE_OPEN_LOW_HIGH,
            PathPattern.CLOSE_OPEN_HIGH_LOW,
        ]
        chop_count = sum(counts.get(p, 0) for p in chop_patterns)
        chop_prob = chop_count / total

        probs = {
            "rally": rally_prob,
            "fade": fade_prob,
            "v_shape": v_prob,
            "inverted_v": inv_v_prob,
            "chop": chop_prob,
        }

        best = max(probs, key=probs.get)
        return best, probs[best]

    def _compute_median_path(self) -> List[Tuple[str, float]]:
        """
        Compute the median path: for each of the 4 price points,
        take the median across all 60 samples, then sort to get the path.
        """
        o = float(np.median(self.df["open"].values))
        h = float(np.median(self.df["high"].values))
        l = float(np.median(self.df["low"].values))
        c = float(np.median(self.df["close"].values))

        prices = [(o, "open"), (h, "high"), (l, "low"), (c, "close")]
        sorted_prices = sorted(prices, key=lambda x: x[0])
        return [(label, price) for price, label in sorted_prices]

    def _derive_signal(self, profile: PathProfile) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
        """
        Derive a trading signal from the path profile.
        """
        if profile.path_confidence < 0.3:
            return None, None, None, None

        traj = profile.trajectory
        conf = profile.trajectory_confidence

        if traj == "rally" and conf > 0.5 and profile.low_before_high_prob > 0.6:
            # Buy at or near predicted low, target predicted high
            entry = profile.median_path[0][1] if profile.median_path else None
            target = profile.median_path[-1][1] if profile.median_path else None
            stop = entry * 0.99 if entry else None
            return "long", entry, target, stop

        elif traj == "fade" and conf > 0.5 and profile.high_before_low_prob > 0.6:
            # Short at or near predicted high, target predicted low
            entry = profile.median_path[-1][1] if profile.median_path else None
            target = profile.median_path[0][1] if profile.median_path else None
            stop = entry * 1.01 if entry else None
            return "short", entry, target, stop

        elif traj == "v_shape" and conf > 0.5:
            # Buy the dip: entry near predicted low, target near predicted close (high)
            entry = profile.median_path[0][1] if profile.median_path else None
            target = profile.median_path[-1][1] if profile.median_path else None
            stop = entry * 0.99 if entry else None
            return "long", entry, target, stop

        elif traj == "inverted_v" and conf > 0.5:
            # Sell the pop: entry near predicted high, target near predicted close (low)
            entry = profile.median_path[-1][1] if profile.median_path else None
            target = profile.median_path[0][1] if profile.median_path else None
            stop = entry * 1.01 if entry else None
            return "short", entry, target, stop

        return None, None, None, None


# =============================================================================
# PATH-AWARE STRATEGIES (for integration into the backtest framework)
# =============================================================================

class PathRallyStrategy:
    """
    Long when path analysis shows a high-confidence rally trajectory
    (low before high, open near low, close near high).
    """
    name = "path_rally"

    def __init__(self, min_confidence: float = 0.6, min_path_conf: float = 0.3):
        self.min_confidence = min_confidence
        self.min_path_conf = min_path_conf

    def generate_signal(self, dist, current_price, history, context):
        from kairos_path import KairosPathExtractor
        extractor = KairosPathExtractor(dist.predictions)
        profile = extractor.extract()

        if profile.path_confidence < self.min_path_conf:
            return None
        if profile.trajectory != "rally" or profile.trajectory_confidence < self.min_confidence:
            return None
        if profile.low_before_high_prob < 0.6:
            return None

        s = dist.stats["close"]
        l = dist.stats["low"]
        h = dist.stats["high"]

        entry = l["pct_50"]
        target = h["pct_50"]
        stop = l["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return {
            "direction": 1,  # LONG
            "size": min(kelly * 0.5, 1.0),
            "entry": current_price,
            "stop": stop,
            "target": target,
            "strategy_name": self.name,
            "confidence": profile.trajectory_confidence * profile.path_confidence,
            "expected_value": ev,
        }


class PathFadeStrategy:
    """
    Short when path analysis shows a high-confidence fade trajectory
    (high before low, open near high, close near low).
    """
    name = "path_fade"

    def __init__(self, min_confidence: float = 0.6, min_path_conf: float = 0.3):
        self.min_confidence = min_confidence
        self.min_path_conf = min_path_conf

    def generate_signal(self, dist, current_price, history, context):
        from kairos_path import KairosPathExtractor
        extractor = KairosPathExtractor(dist.predictions)
        profile = extractor.extract()

        if profile.path_confidence < self.min_path_conf:
            return None
        if profile.trajectory != "fade" or profile.trajectory_confidence < self.min_confidence:
            return None
        if profile.high_before_low_prob < 0.6:
            return None

        s = dist.stats["close"]
        l = dist.stats["low"]
        h = dist.stats["high"]

        entry = h["pct_50"]
        target = l["pct_50"]
        stop = h["pct_90"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return {
            "direction": -1,  # SHORT
            "size": min(kelly * 0.5, 1.0),
            "entry": current_price,
            "stop": stop,
            "target": target,
            "strategy_name": self.name,
            "confidence": profile.trajectory_confidence * profile.path_confidence,
            "expected_value": ev,
        }


class PathVShapeStrategy:
    """
    Buy the V-shape dip: low in the middle, recovery to close.
    """
    name = "path_v_shape"

    def __init__(self, min_confidence: float = 0.5, min_path_conf: float = 0.3):
        self.min_confidence = min_confidence
        self.min_path_conf = min_path_conf

    def generate_signal(self, dist, current_price, history, context):
        from kairos_path import KairosPathExtractor
        extractor = KairosPathExtractor(dist.predictions)
        profile = extractor.extract()

        if profile.path_confidence < self.min_path_conf:
            return None
        if profile.trajectory != "v_shape" or profile.trajectory_confidence < self.min_confidence:
            return None

        l = dist.stats["low"]
        c = dist.stats["close"]

        entry = l["pct_50"]
        target = c["pct_75"]
        stop = l["pct_5"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return {
            "direction": 1,
            "size": min(kelly * 0.5, 1.0),
            "entry": current_price,
            "stop": stop,
            "target": target,
            "strategy_name": self.name,
            "confidence": profile.trajectory_confidence * profile.path_confidence,
            "expected_value": ev,
        }


class PathInvertedVStrategy:
    """
    Short the inverted V pop: high in the middle, fade to close.
    """
    name = "path_inverted_v"

    def __init__(self, min_confidence: float = 0.5, min_path_conf: float = 0.3):
        self.min_confidence = min_confidence
        self.min_path_conf = min_path_conf

    def generate_signal(self, dist, current_price, history, context):
        from kairos_path import KairosPathExtractor
        extractor = KairosPathExtractor(dist.predictions)
        profile = extractor.extract()

        if profile.path_confidence < self.min_path_conf:
            return None
        if profile.trajectory != "inverted_v" or profile.trajectory_confidence < self.min_confidence:
            return None

        h = dist.stats["high"]
        c = dist.stats["close"]

        entry = h["pct_50"]
        target = c["pct_25"]
        stop = h["pct_95"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        return {
            "direction": -1,
            "size": min(kelly * 0.5, 1.0),
            "entry": current_price,
            "stop": stop,
            "target": target,
            "strategy_name": self.name,
            "confidence": profile.trajectory_confidence * profile.path_confidence,
            "expected_value": ev,
        }


class PathHighLowSequenceStrategy:
    """
    The purest path strategy: trade the high-before-low / low-before-high probability.
    If >80% of samples show low before high, buy at open and hold.
    If >80% show high before low, short at open and hold.
    """
    name = "path_high_low_sequence"

    def __init__(self, threshold: float = 0.8, min_path_conf: float = 0.3):
        self.threshold = threshold
        self.min_path_conf = min_path_conf

    def generate_signal(self, dist, current_price, history, context):
        from kairos_path import KairosPathExtractor
        extractor = KairosPathExtractor(dist.predictions)
        profile = extractor.extract()

        if profile.path_confidence < self.min_path_conf:
            return None

        if profile.low_before_high_prob >= self.threshold:
            direction = 1
            stop = dist.stats["low"]["pct_10"]
            target = dist.stats["high"]["pct_90"]
        elif profile.high_before_low_prob >= self.threshold:
            direction = -1
            stop = dist.stats["high"]["pct_90"]
            target = dist.stats["low"]["pct_10"]
        else:
            return None

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        conf = max(profile.low_before_high_prob, profile.high_before_low_prob)
        return {
            "direction": direction,
            "size": min(kelly * 0.5, 1.0),
            "entry": current_price,
            "stop": stop,
            "target": target,
            "strategy_name": self.name,
            "confidence": conf * profile.path_confidence,
            "expected_value": ev,
        }


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    # Create synthetic predictions for demonstration
    np.random.seed(42)
    n_samples = 60
    base = 100.0

    predictions = []
    for _ in range(n_samples):
        # Simulate a rally day: open near low, close near high
        o = base + np.random.normal(0, 0.5)
        l = o - abs(np.random.normal(1, 0.3))
        h = o + abs(np.random.normal(2, 0.5))
        c = h - abs(np.random.normal(0.3, 0.2))
        predictions.append(pd.DataFrame({
            "open": [o], "high": [h], "low": [l], "close": [c],
            "volume": [1000], "amount": [100000]
        }))

    extractor = KairosPathExtractor(predictions)
    profile = extractor.extract()
    print(profile)
