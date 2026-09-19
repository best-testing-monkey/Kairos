#!/usr/bin/env python3
"""E18-S03: Train per-candidate GBM classifiers with purged CV.

Joins E18-S01's hindsight labels (`tpsl_label_candidates`) with E18-S02's
price-history features (`kairos_tpsl_features.extract_features`) and trains
one `GradientBoostedStumps` (`strategy/kairos_ml.py`, numpy-only -- no new ML
dependency) per (stop_pct, target_pct) grid candidate, predicting
`P(target hit before stop | features)`.

Purged/embargoed time split (not random -- see `purged_time_split()`'s
docstring for the AFML citation and the documented default) and a per-class
data-sufficiency check (see `check_class_sufficiency()`) decide whether to
train one pooled model per candidate or one model per (candidate, asset
class). Trained classifiers are pickled to `data/tpsl_models/`, alongside a
shared `feature_metadata.json` recording the exact feature-vector column
order -- E18-S04's inference wrapper MUST reproduce this order (see the
comment on `_feature_columns()` below).

Usage:
    uv run scripts/train_tpsl_model.py [--db PATH] [--output-dir PATH]
        [--val-fraction 0.2] [--embargo-bars 5] [--min-signals 30]
        [--lookback-days 120] [--limit N]
"""
import argparse
import json
import os
import pickle
import sqlite3
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import price_cache  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))

from kairos_ml import GradientBoostedStumps  # type: ignore  # noqa: E402
from kairos_tpsl_features import extract_features  # type: ignore  # noqa: E402
from kairos_signal_replay import _ensure_configured_db  # type: ignore  # noqa: E402
from kairos_pipeline import CLASS_STATS_MIN_SIGNALS  # type: ignore  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_results.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tpsl_models")

# Numeric columns straight from kairos_tpsl_features.extract_features()'s return dict.
NUMERIC_FEATURES = ("atr", "realized_vol", "trend_10", "range_position")

# kairos_strategies.asset_class_for()'s docstring guarantees exactly these five
# values -- a fixed, documented contract (unlike the other classifiers this repo's
# CLAUDE.md warns not to treat as interchangeable), so hardcoding the vocabulary
# here is safe.
ASSET_CLASSES = ("equity", "crypto", "fx", "commodity", "mixed")


def purged_time_split(
    as_of_values: Any,
    val_fraction: float = 0.2,
    embargo_bars: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered train/validation split with an embargoed gap.

    Design decision (E18-S03 -- the ticket flags this as the one part of the
    story needing real judgment, not mechanical copying). AFML (Lopez de
    Prado) warns that a *random* train/val split leaks adjacent-in-time
    volatility-regime information for exactly this parameter class
    (stop/target barrier width): nearby-in-time signals share a regime, so a
    random split lets validation rows "see" information correlated with
    their training-set neighbors. Purged/embargoed CV fixes this by (a)
    splitting on time instead of randomly, and (b) dropping a small gap of
    rows immediately before the validation cutoff so no training row sits
    within one embargo window of a validation row.

    Defaults are a documented, reasonable choice, not a calibrated one --
    picked per the ticket's own instruction ("if anything about the embargo
    window size is unclear, pick a documented, reasonable default and flag
    it in a comment rather than guessing silently"):
      - val_fraction=0.2: last 20% of signals by time held out -- the
        ticket's own example split.
      - embargo_bars=5: "bars" here means rows in the as_of-sorted sequence,
        not a fixed wall-clock duration -- signals here are pooled across
        tickers/intervals with irregular timestamps (unlike a single
        instrument's regular bar grid), so there is no one bar-duration to
        embargo by calendar time. 5 rows is the ticket's own suggested
        example size for this exact parameter class.

    Args:
        as_of_values: signal timestamps, any type pandas.to_datetime accepts.
        val_fraction: fraction of rows (by time) held out as validation.
        embargo_bars: rows immediately before the validation cutoff dropped
            from training (in neither the train nor the validation set).

    Returns:
        (train_idx, val_idx): integer index arrays into the *original*
        as_of_values sequence, sorted ascending by time within each. Embargoed
        rows appear in neither array.
    """
    n = len(as_of_values)
    order = np.argsort(pd.to_datetime(list(as_of_values)).values, kind="stable")

    n_val = min(n, max(1, int(round(n * val_fraction)))) if n else 0
    cutoff = n - n_val
    embargo_start = max(0, cutoff - embargo_bars)

    return order[:embargo_start], order[cutoff:]


def check_class_sufficiency(
    asset_classes: Any,
    min_signals: int = CLASS_STATS_MIN_SIGNALS,
) -> tuple[bool, dict[str, int]]:
    """Per-class signal-count sufficiency check for one candidate's training set.

    Mirrors the fallback-to-corpus precedent in
    `kairos_pipeline.strategy_class_stats()` (`CLASS_STATS_MIN_SIGNALS` --
    "enough signals to trust a class-level statistic"). Returns
    `(per_class_ok, counts)`: `per_class_ok` is False if the whole set, or
    any single class within it, falls below `min_signals` -- the caller
    should then fall back to one pooled-across-classes model rather than a
    fragile per-class one.

    Per this repo's tracked project state the deduped signal corpus skews
    heavily equity, so this is *expected* to return False for crypto/fx in
    practice -- that is the check doing its job, not a bug to fix by
    lowering the threshold.
    """
    counts: dict[str, int] = dict(Counter(asset_classes))
    total = sum(counts.values())
    if total < min_signals:
        return False, counts
    if any(c < min_signals for c in counts.values()):
        return False, counts
    return True, counts


def _feature_columns(interval_vocab: list[str]) -> list[str]:
    """Deterministic feature-vector column order.

    IMPORTANT -- silent-failure risk called out explicitly in the ticket:
    this exact order is written to feature_metadata.json's "feature_columns"
    key, and every classifier saved by this script was trained against
    columns in this order. E18-S04's inference wrapper MUST vectorize its
    own input features in this identical order before calling a loaded
    classifier's predict_proba() -- a mismatched feature order/set between
    training and inference is a silent-failure risk (wrong columns score
    without raising), not a loud one.
    """
    cols = list(NUMERIC_FEATURES)
    cols += [f"asset_class={c}" for c in ASSET_CLASSES]
    cols += [f"interval={iv}" for iv in interval_vocab]
    return cols


def _vectorize(features_list: list[dict[str, Any]], interval_vocab: list[str]) -> np.ndarray:
    """Turn a list of extract_features() dicts into a numeric matrix.

    Categorical features (asset_class, interval) are one-hot encoded --
    GradientBoostedStumps only supports numeric threshold splits. Column
    order matches _feature_columns(interval_vocab) exactly (see its
    docstring for why that matters).
    """
    columns = _feature_columns(interval_vocab)
    X = np.zeros((len(features_list), len(columns)), dtype=float)
    for i, feats in enumerate(features_list):
        for j, name in enumerate(NUMERIC_FEATURES):
            X[i, j] = feats[name]
        ac_col = len(NUMERIC_FEATURES) + ASSET_CLASSES.index(feats["asset_class"])
        X[i, ac_col] = 1.0
        if feats["interval"] in interval_vocab:
            iv_col = len(NUMERIC_FEATURES) + len(ASSET_CLASSES) + interval_vocab.index(feats["interval"])
            X[i, iv_col] = 1.0
    return X


def _fetch_history(
    ticker: str,
    as_of: str,
    interval: str,
    db_path: str,
    lookback_days: int,
) -> pd.DataFrame | None:
    """Fetch OHLCV history ending at as_of, lowercased for extract_features().

    Re-fetches per signal the same way E18-S02's extract_features() docstring
    describes: fetch a backward window, let extract_features() truncate to
    as_of internally (its own no-lookahead guarantee), rather than trying to
    pre-truncate here.
    """
    as_of_ts = pd.Timestamp(as_of)
    start = (as_of_ts - pd.Timedelta(days=lookback_days)).date().isoformat()
    end = as_of_ts.date().isoformat()
    try:
        bars = price_cache.get_price_data(
            ticker, start_date=start, end_date=end, interval=interval, db_path=db_path
        )
    except Exception:
        return None
    if bars is None or bars.empty:
        return None
    return bars.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]


def _extract_all_features(
    unique_signals: pd.DataFrame,
    db_path: str,
    lookback_days: int,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Compute extract_features() once per unique signal_id.

    Features don't depend on the (stop_pct, target_pct) grid -- only on the
    signal's own ticker/as_of/interval/entry -- so this is done once per
    signal, not once per (signal, candidate) row, then reused across all
    grid candidates that signal appears in.
    """
    _ensure_configured_db(db_path)
    features_by_signal: dict[str, dict[str, Any]] = {}
    intervals_seen: set[str] = set()
    n_skipped = 0
    for row in unique_signals.itertuples():
        # itertuples() yields Any-typed fields per pandas' stubs; this data is
        # sourced from TEXT/REAL sqlite columns, so the str()/float() casts
        # below are just narrowing for mypy, not behavior changes.
        ticker, as_of, interval = str(row.ticker), str(row.as_of), str(row.interval)
        signal_id = str(row.signal_id)
        entry = float(row.entry)  # type: ignore[arg-type]
        hist = _fetch_history(ticker, as_of, interval, db_path, lookback_days)
        if hist is None:
            n_skipped += 1
            continue
        try:
            feats = extract_features(ticker, as_of, interval, entry, hist)
        except ValueError:
            n_skipped += 1
            continue
        features_by_signal[signal_id] = feats
        intervals_seen.add(feats["interval"])
    if n_skipped:
        print(
            f"[warn] skipped {n_skipped}/{len(unique_signals)} signals: no price history or "
            f"insufficient bars for feature extraction"
        )
    return features_by_signal, sorted(intervals_seen)


def _fit_with_purged_split(
    X: np.ndarray,
    y: np.ndarray,
    as_of_values: list[str],
    val_fraction: float,
    embargo_bars: int,
) -> tuple[GradientBoostedStumps, dict[str, Any]]:
    train_idx, val_idx = purged_time_split(as_of_values, val_fraction, embargo_bars)
    if len(train_idx) == 0:
        # Too little data to both split and embargo -- train on everything
        # rather than fail; this candidate's own signal count is already
        # logged above (see the caller), so a small-data outcome is visible.
        train_idx = np.arange(len(y))
        val_idx = np.array([], dtype=int)

    model = GradientBoostedStumps()
    model.fit(X[train_idx], y[train_idx])

    val_stats: dict[str, Any] = {"n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
    if len(val_idx) > 0:
        val_pred = model.predict_proba(X[val_idx])
        val_stats["val_accuracy"] = float(np.mean((val_pred >= 0.5) == (y[val_idx] >= 0.5)))
    return model, val_stats


def train_and_save(
    conn: sqlite3.Connection,
    db_path: str,
    output_dir: str,
    *,
    val_fraction: float = 0.2,
    embargo_bars: int = 5,
    min_signals: int = CLASS_STATS_MIN_SIGNALS,
    lookback_days: int = 120,
    limit: int | None = None,
) -> dict[str, Any]:
    """Join labels+features, train one GBM per grid candidate, save to disk."""
    try:
        candidates = pd.read_sql_query(
            """
            SELECT c.signal_id, c.stop_pct, c.target_pct, c.hit_target_first,
                   s.ticker, s.direction, s.interval, s.as_of, s.entry
            FROM tpsl_label_candidates c
            JOIN papertrade_signals s ON c.signal_id = s.signal_id
            WHERE c.resolved = 1 AND c.hit_target_first IS NOT NULL
            """,
            conn,
        )
    except (pd.errors.DatabaseError, sqlite3.OperationalError) as exc:
        print(f"[warn] tpsl_label_candidates unreadable ({exc}); run scripts/tpsl_label_grid.py first")
        return {"trained": 0, "reason": "no_table"}

    if candidates.empty:
        print("[warn] no resolved tpsl_label_candidates rows found; nothing to train")
        return {"trained": 0, "reason": "no_data"}

    unique_signals = candidates.drop_duplicates("signal_id")
    if limit:
        unique_signals = unique_signals.head(limit)
        candidates = candidates[candidates["signal_id"].isin(set(unique_signals["signal_id"]))]

    features_by_signal, interval_vocab = _extract_all_features(unique_signals, db_path, lookback_days)
    if not features_by_signal:
        print("[warn] no signal had sufficient history for feature extraction")
        return {"trained": 0, "reason": "no_features"}

    joined = candidates[candidates["signal_id"].isin(set(features_by_signal))]
    dropped = len(candidates) - len(joined)
    if dropped:
        print(f"[info] dropped {dropped} candidate rows: signal lacked extractable features")

    os.makedirs(output_dir, exist_ok=True)
    feature_columns = _feature_columns(interval_vocab)

    combo_df = joined[["stop_pct", "target_pct"]].drop_duplicates().sort_values(["stop_pct", "target_pct"])
    grid = list(combo_df.itertuples(index=False, name=None))

    trained_meta: dict[str, Any] = {}
    for stop_pct, target_pct in grid:
        sub = joined[(joined["stop_pct"] == stop_pct) & (joined["target_pct"] == target_pct)]
        feats_list = [features_by_signal[sid] for sid in sub["signal_id"]]
        X = _vectorize(feats_list, interval_vocab)
        y = sub["hit_target_first"].to_numpy(dtype=float)
        asset_classes = [f["asset_class"] for f in feats_list]
        as_of_values = sub["as_of"].tolist()

        # Sufficiency check is per-candidate (not computed once globally):
        # different (stop_pct, target_pct) widths resolve/disqualify
        # different subsets of signals (see tpsl_label_grid.py), so each
        # candidate's own trainable set can have a different class mix.
        per_class_ok, class_counts = check_class_sufficiency(asset_classes, min_signals)
        key = f"{stop_pct}_{target_pct}"
        print(f"[info] candidate {key}: n={len(y)} class_counts={class_counts} per_class_ok={per_class_ok}")

        if per_class_ok:
            for cls in sorted(set(asset_classes)):
                mask = np.array([c == cls for c in asset_classes])
                model, val_stats = _fit_with_purged_split(
                    X[mask], y[mask], [a for a, m in zip(as_of_values, mask) if m],
                    val_fraction, embargo_bars,
                )
                entry_key = f"{key}_{cls}"
                with open(os.path.join(output_dir, f"{entry_key}.pkl"), "wb") as fh:
                    pickle.dump(model, fh)
                trained_meta[entry_key] = {
                    "stop_pct": stop_pct, "target_pct": target_pct, "asset_class": cls,
                    "mode": "per_class", "class_counts": class_counts, **val_stats,
                }
        else:
            print(f"[warn] candidate {key}: below min_signals={min_signals} for per-class training; "
                  f"falling back to pooled model")
            model, val_stats = _fit_with_purged_split(X, y, as_of_values, val_fraction, embargo_bars)
            with open(os.path.join(output_dir, f"{key}.pkl"), "wb") as fh:
                pickle.dump(model, fh)
            trained_meta[key] = {
                "stop_pct": stop_pct, "target_pct": target_pct, "mode": "pooled",
                "class_counts": class_counts, **val_stats,
            }

    metadata_path = os.path.join(output_dir, "feature_metadata.json")
    with open(metadata_path, "w") as fh:
        json.dump(
            {
                "feature_columns": feature_columns,
                "asset_classes": list(ASSET_CLASSES),
                "interval_vocab": interval_vocab,
                "val_fraction": val_fraction,
                "embargo_bars": embargo_bars,
                "min_signals": min_signals,
                "candidates": trained_meta,
            },
            fh,
            indent=2,
        )

    return {"trained": len(trained_meta), "output_dir": output_dir, "metadata_path": metadata_path}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH, help="Path to pipeline_results.db")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to write trained models + metadata")
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Fraction of signals (by time) held out as purged validation set")
    ap.add_argument("--embargo-bars", type=int, default=5,
                    help="Rows immediately before the validation cutoff dropped from training")
    ap.add_argument("--min-signals", type=int, default=CLASS_STATS_MIN_SIGNALS,
                    help="Per-class signal-count threshold below which training falls back to pooled")
    ap.add_argument("--lookback-days", type=int, default=120,
                    help="Calendar days of history fetched per signal for feature extraction")
    ap.add_argument("--limit", type=int, default=None, help="Limit unique signals processed (for testing)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        summary = train_and_save(
            conn, args.db, args.output_dir,
            val_fraction=args.val_fraction, embargo_bars=args.embargo_bars,
            min_signals=args.min_signals, lookback_days=args.lookback_days,
            limit=args.limit,
        )
    finally:
        conn.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
