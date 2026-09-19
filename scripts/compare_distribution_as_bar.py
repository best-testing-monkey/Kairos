#!/usr/bin/env python3
"""E20-S05: Offline comparison -- distribution_as_bar vs. last_real_bar.

Compares the two indicator-gated strategies E20-S04 confirmed are actually
sensitive to `distribution_as_bar` (RSIFilterStrategy, MACDFilterStrategy --
see tests/unit/test_strategy_signals.py's TestIndicatorSensitivityToSyntheticBar)
over the same historical window, once per `OrchestratorConfig.prediction_usage_mode`,
and reports signal-count/EV/Sharpe deltas. This is the gate before considering
distribution_as_bar for live use, per
docs/tickets/DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md section 7.

IMPORTANT -- how this differs from E18-S05 (scripts/compare_tpsl_model.py):
E18-S05 resolves an ALREADY-DECIDED signal's outcome against a different
stop/target candidate (MLBracketStrategy) -- it never re-runs signal
generation. This story re-runs signal GENERATION itself, twice, via
KairosOrchestrator.run_backtest() (which drives _run_day() per date), because
distribution_as_bar changes what generate_signal() decides in the first place
(the synthetic bar reaches RSI/MACD's own indicator math over `history`), not
just how an already-decided signal's stop/target gets resolved afterward. Do
not follow E18-S05's "resolve existing candidates" shape here -- it doesn't
apply.

Uses oracle mode (no_prediction=True) so this needs no GPU/live model,
matching this module's own no-GPU/no-network testability convention
(docs/tickets/APPENDIX-A-standards.md).

NOTE: Like E18-S05, this is a directional comparison, not a live-P&L claim
(see DESIGN_DOC_offline_signal_replay.md's cost-model caveat).

Usage:
    uv run scripts/compare_distribution_as_bar.py [--db PATH] [--output-dir PATH]
        [--symbols SYM1,SYM2,...] [--start DATE] [--end DATE] [--lookback N]
"""
import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))

from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig, StrategyRegistry  # type: ignore  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_results.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "distribution_as_bar_validation")

# E20-S04 confirmed these are the only strategies whose generate_signal() reads
# `history` for indicator math (RSI/MACD) and is therefore sensitive to the
# synthetic bar distribution_as_bar appends -- every other strategy either
# reads dist.stats directly (TrendFollowingStrategy and its 7 aliases, per
# this repo's "Eight strategy names are the same strategy" note) or doesn't
# touch history at all. Scoping to just these two keeps signal-count/EV/Sharpe
# deltas attributable to them, instead of diluted across ~100 unrelated
# strategies that this mode has zero effect on by construction (design doc
# section 3).
SCOPED_STRATEGIES = ("rsi_filter", "macd_filter")

_PREDICTION_USAGE_MODES = ("last_real_bar", "distribution_as_bar")


def _scoped_disabled_strategies() -> set:
    """All registered strategy names except SCOPED_STRATEGIES.

    Reuses OrchestratorConfig.disabled_strategies (subtractive gate, applied
    in StrategyRegistry.build_all before Kurtosis/LiquidityFilter wrapping)
    to isolate the two strategies this comparison cares about.

    Built from an UNWRAPPED probe registry (kurtosis_action="none",
    min_volume_percentile=0) because the disabled-strategies filter runs
    against each strategy's raw .name *before* wrapping -- using a wrapped
    list would collect the same names (wrappers preserve the inner .name)
    but at needless extra cost, so the probe skips wrapping outright.
    """
    probe_cfg = OrchestratorConfig(disabled_strategies=set(), kurtosis_action="none", min_volume_percentile=0.0)
    all_names = {s.name for s in StrategyRegistry.build_all(probe_cfg)}
    return all_names - set(SCOPED_STRATEGIES)


def run_scoped_backtest(
    data_dict: dict[str, pd.DataFrame], assets: list, prediction_usage_mode: str, lookback: int = 30
) -> dict:
    """Run one full KairosOrchestrator.run_backtest() scoped to SCOPED_STRATEGIES.

    Oracle mode (no_prediction=True, naive_baseline=False): distribution comes
    from the real next bar (a deliberate future peek -- the perfect-foresight
    ceiling every other oracle_results row in this codebase uses), not a live
    model call, so this needs no GPU and is deterministic given `data_dict`.
    """
    config = OrchestratorConfig(
        no_prediction=True,
        naive_baseline=False,
        prediction_usage_mode=prediction_usage_mode,
        disabled_strategies=_scoped_disabled_strategies(),
    )
    orch = KairosOrchestrator(predict_fn=lambda *a, **kw: [], assets=assets, config=config)
    return orch.run_backtest(data_dict, lookback=lookback)


def compare_prediction_usage_modes(data_dict: dict[str, pd.DataFrame], assets: list, lookback: int = 30) -> dict:
    """Compare SCOPED_STRATEGIES' shadow performance under both prediction usage modes.

    Runs run_backtest() twice over the SAME data_dict/window -- once per mode
    in _PREDICTION_USAGE_MODES -- and diffs each scoped strategy's
    signal_count/EV/Sharpe between the two runs.

    This compares signal-GENERATION behavior (distribution_as_bar can change
    what generate_signal() decides, via the synthetic bar reaching RSI/MACD's
    indicator math), not exit resolution of an already-decided signal -- see
    this module's docstring for the full distinction from E18-S05's
    compare_tpsl_model.py, whose comparison shape does not apply here.
    """
    results_by_mode = {
        mode: run_scoped_backtest(data_dict, assets, mode, lookback=lookback)
        for mode in _PREDICTION_USAGE_MODES
    }

    strategies_report: dict[str, Any] = {}
    for strat in SCOPED_STRATEGIES:
        per_mode = {}
        for mode in _PREDICTION_USAGE_MODES:
            sd = results_by_mode[mode]["shadow_performance"].get(
                strat, {"pnl_list": [], "sharpe": 0.0, "signal_count": 0}
            )
            pnl_list = sd.get("pnl_list", [])
            per_mode[mode] = {
                "signal_count": sd.get("signal_count", 0),
                "ev_pct": float(np.mean(pnl_list)) * 100.0 if pnl_list else 0.0,
                "sharpe": sd.get("sharpe", 0.0),
            }
        base, synth = per_mode["last_real_bar"], per_mode["distribution_as_bar"]
        strategies_report[strat] = {
            "last_real_bar": base,
            "distribution_as_bar": synth,
            "delta": {
                "signal_count": synth["signal_count"] - base["signal_count"],
                "ev_pct": synth["ev_pct"] - base["ev_pct"],
                "sharpe": synth["sharpe"] - base["sharpe"],
            },
        }

    return {
        "scoped_strategies": list(SCOPED_STRATEGIES),
        "strategies": strategies_report,
        "comparison_note": (
            "This report compares signal-GENERATION behavior (KairosOrchestrator."
            "run_backtest() re-run once per prediction_usage_mode over the same "
            "window), not exit resolution of an already-decided signal -- unlike "
            "E18-S05's compare_tpsl_model.py, which resolves fixed candidates "
            "against different stop/target choices. distribution_as_bar changes "
            "what generate_signal() itself decides, so re-running generation is "
            "required."
        ),
        "directional_comparison_note": (
            "This is a directional comparison, not a live-P&L claim. The cost "
            "model diverges from phantom's per-instrument model (see "
            "DESIGN_DOC_offline_signal_replay.md)."
        ),
    }


def _fetch_data_dict(symbols: list, start: str, end: str, interval: str, db_path: str) -> dict:
    """Fetch OHLCV history for each symbol via price_cache, lower-cased columns
    (open/high/low/close/volume) to match AssetPrediction/history conventions
    used throughout strategy/kairos_orchestrator.py.
    """
    import price_cache  # type: ignore
    price_cache.configure(remote=False, local_mirror_path=db_path)

    data_dict = {}
    for symbol in symbols:
        bars = price_cache.get_price_data(symbol, start_date=start, end_date=end,
                                          interval=interval, db_path=db_path)
        if bars is None or bars.empty:
            print(f"[warn] no data for {symbol}, skipping")
            continue
        data_dict[symbol] = bars.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]
    return data_dict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH, help="Path to pipeline_results.db")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to write comparison report")
    ap.add_argument("--symbols", default="AAPL,MSFT", help="Comma-separated symbol list")
    ap.add_argument("--start", required=True, help="Window start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="Window end date (YYYY-MM-DD)")
    ap.add_argument("--interval", default="1d", help="Bar interval")
    ap.add_argument("--lookback", type=int, default=30, help="Bars of history required before evaluating a day")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"[info] Fetching {symbols} {args.interval} bars [{args.start}, {args.end}]...")
    data_dict = _fetch_data_dict(symbols, args.start, args.end, args.interval, args.db)
    if not data_dict:
        raise SystemExit("No data fetched for any symbol; aborting")

    print("[info] Running scoped comparison (rsi_filter, macd_filter)...")
    report = compare_prediction_usage_modes(data_dict, list(data_dict.keys()), lookback=args.lookback)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "distribution_as_bar_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[info] Report written to {report_path}")
    print("\n=== distribution_as_bar vs. last_real_bar ===")
    print(report["comparison_note"])
    for strat, data in report["strategies"].items():
        print(f"\n{strat}:")
        print(f"  last_real_bar:       {data['last_real_bar']}")
        print(f"  distribution_as_bar: {data['distribution_as_bar']}")
        print(f"  delta:               {data['delta']}")
    print(f"\nNote: {report['directional_comparison_note']}")


if __name__ == "__main__":
    main()
