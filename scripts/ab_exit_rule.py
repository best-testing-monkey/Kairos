"""A/B the old one-bar-force-close exit rule against the new multi-bar walk.

One oracle run -> one set of shadow signals -> both rules scored on identical
inputs. Measures exactly what the methodology change costs and buys.
"""
import os, sys, json
from collections import defaultdict
import numpy as np

REPO = "/media/baz/MonkeyWorks/PycharmProjects/Kairos"
sys.path.insert(0, os.path.join(REPO, "strategy"))
os.chdir(REPO)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import kairos_strategies as ks
from kairos_strategies import KairosSettings, fetch_data_raw, predict_all_batch, predict_kairos_cloud
from kairos_orchestrator import (KairosOrchestrator, OrchestratorConfig, _summarize_pnl,
                                 _resolve_exit, bars_per_year)
from kairos_backtest import Direction, asset_class_of_symbol

ASSETS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["AAPL", "MSFT", "NVDA", "AMD"]
INTERVAL, PERIOD = "1d", "6m"

KairosSettings.assets = ASSETS
KairosSettings.interval = INTERVAL
KairosSettings.backtest_period = PERIOD
KairosSettings.no_prediction = True
KairosSettings.naive_baseline = False
KairosSettings.pred_samples = 100

nbars = ks._period_to_bars(PERIOD, INTERVAL)
cfg = OrchestratorConfig.for_interval(
    INTERVAL, initial_capital=KairosSettings.initial_capital,
    cross_asset_ranking=True, online_weighting=True, partial_exits=True,
    max_horizon=3, no_prediction=True, naive_baseline=False, disabled_strategies=set(),
)
orch = KairosOrchestrator(
    predict_fn=predict_kairos_cloud, assets=ASSETS, config=cfg,
    batch_predict_fn=predict_all_batch, model=KairosSettings.model,
    tokenizer=KairosSettings.tokenizer, symbol=KairosSettings.symbol,
)
lookback = KairosSettings.lookback
orch.run_backtest({s: fetch_data_raw(s, lookback, min_bars=lookback + nbars).tail(lookback + nbars)
                   for s in ASSETS}, lookback=lookback)


def score(rule):
    by = defaultdict(list)
    for rec in orch._shadow_signals:
        date, symbol, sname, direction, stop, target, sig_entry = rec
        df = orch._data_dict.get(symbol)
        if df is None:
            continue
        future = df[df.index > date]
        if future.empty:
            continue
        entry = float(future.iloc[0]["open"])
        if entry <= 0:
            continue
        ref = float(sig_entry) if sig_entry and sig_entry > 0 else entry
        sp = entry * (1.0 + (stop - ref) / ref)
        tp = entry * (1.0 + (target - ref) / ref)
        ex = rule(future, direction, sp, tp)
        if ex is None:
            continue
        by[sname].append((ex - entry) / entry if direction == Direction.LONG
                         else (entry - ex) / entry)
    ann = np.sqrt(bars_per_year(INTERVAL))
    return {k: _summarize_pnl(v, ann) for k, v in by.items()}


def old_rule(future, direction, sp, tp):
    """One bar, force-close at its close."""
    nb = future.iloc[0]
    h, l, c = float(nb["high"]), float(nb["low"]), float(nb["close"])
    if direction == Direction.LONG:
        return sp if l <= sp else (tp if h >= tp else c)
    return sp if h >= sp else (tp if l <= tp else c)


def new_rule(future, direction, sp, tp):
    return _resolve_exit(future, direction, sp, tp, gap_check_first_bar=False)


old, new = score(old_rule), score(new_rule)

n_old = sum(d["signal_count"] for d in old.values())
n_new = sum(d["signal_count"] for d in new.values())
print(f"\n=== {','.join(ASSETS)} {INTERVAL}/{PERIOD} ===")
print(f"signals: old={n_old}  new={n_new}  dropped={n_old - n_new} ({100*(n_old-n_new)/max(n_old,1):.1f}% unresolved)")
print(f"strategies: old={len(old)} new={len(new)}")

both = sorted(set(old) & set(new))
d_sharpe = [new[k]["sharpe"] - old[k]["sharpe"] for k in both]
print(f"\nsharpe delta over {len(both)} strategies: "
      f"median={np.median(d_sharpe):+.3f} mean={np.mean(d_sharpe):+.3f} "
      f"min={min(d_sharpe):+.3f} max={max(d_sharpe):+.3f}")
print(f"old median sharpe={np.median([old[k]['sharpe'] for k in both]):+.3f}  "
      f"new median sharpe={np.median([new[k]['sharpe'] for k in both]):+.3f}")
sign_flip = sum(1 for k in both if (old[k]["sharpe"] > 0) != (new[k]["sharpe"] > 0))
print(f"sign flips: {sign_flip}/{len(both)}")

print("\nper-strategy (top 12 by |delta|):")
for k in sorted(both, key=lambda k: -abs(new[k]["sharpe"] - old[k]["sharpe"]))[:12]:
    print(f"  {k:<28} sharpe {old[k]['sharpe']:+7.3f} -> {new[k]['sharpe']:+7.3f}   "
          f"n {old[k]['signal_count']:>4} -> {new[k]['signal_count']:>4}")

json.dump({"old": {k: {"sharpe": v["sharpe"], "n": v["signal_count"]} for k, v in old.items()},
           "new": {k: {"sharpe": v["sharpe"], "n": v["signal_count"]} for k, v in new.items()}},
          open("/tmp/claude-1000/-media-baz-MonkeyWorks-PycharmProjects-Kairos/5eb09df8-e498-47ee-82c6-83730d77ec7c/scratchpad/ab_result.json", "w"), indent=1)
