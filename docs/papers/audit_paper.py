#!/usr/bin/env python3
"""Independently recompute every headline number in The Prediction Premium
from the database and check it against the published HTML.

Deliberately does NOT reuse paper_table.json or build_paper.py's constants --
it re-derives from oracle_results/model_results so a transcription or staleness
error in the generator would show up as a mismatch rather than agreeing with
itself.
"""
import sqlite3, statistics as st, html, re, pathlib, sys

DB = "/media/baz/MonkeyWorks/PycharmProjects/Kairos/data/pipeline_results.db"
PAGE = pathlib.Path("/media/baz/MonkeyWorks/PycharmProjects/Kairos/docs/papers/prediction_premium.html")
ALIAS = {"cds_spread_filter", "cot_positioning_filter", "dark_pool_filter", "fractal_dimension",
         "gaussian_process", "insider_cluster", "onchain_flow_filter"}  # trend_following kept
# Rows older than the one-exit-rule re-sweep were scored one bar ahead and
# force-closed; they are a different ruler and are excluded everywhere.
CUTOFF = "2026-08-30"

c = sqlite3.connect(DB); c.row_factory = sqlite3.Row

def key_groups(tbl, stage):
    return {(r[0], r[1], r[2]) for r in c.execute(
        f"""SELECT DISTINCT o.assets,o.interval,o.backtest_period FROM {tbl} o
            JOIN runs r ON r.run_id=o.run_id WHERE o.stage=? AND r.timestamp>=?""", (stage, CUTOFF))}

matched = (key_groups("oracle_results", "oracle")
           & key_groups("oracle_results", "naive")
           & key_groups("model_results", "base"))

def stats(tbl, stage):
    agg = {}
    for r in c.execute(f"""SELECT o.strategy_name,o.sharpe,o.signal_count,o.win_rate,o.avg_pnl_per_trade,
                                  o.assets,o.interval,o.backtest_period
                           FROM {tbl} o JOIN runs r ON r.run_id=o.run_id
                           WHERE o.stage=? AND o.signal_count>=3 AND r.timestamp>=?""", (stage, CUTOFF)):
        if (r["assets"], r["interval"], r["backtest_period"]) not in matched: continue
        if r["strategy_name"] in ALIAS: continue
        d = agg.setdefault(r["strategy_name"], {"s": 0, "ws": 0.0, "ww": 0.0, "wp": 0.0})
        n = r["signal_count"]
        d["s"] += n; d["ws"] += r["sharpe"]*n; d["ww"] += r["win_rate"]*n; d["wp"] += r["avg_pnl_per_trade"]*n
    return {k: {"signals": v["s"], "sharpe": v["ws"]/v["s"], "win": v["ww"]/v["s"], "pnl": v["wp"]/v["s"]}
            for k, v in agg.items()}

S = {s: stats(t, s) for t, s in [("oracle_results","oracle"),("oracle_results","naive"),("model_results","base")]}
common = sorted(set(S["oracle"]) & set(S["naive"]) & set(S["base"]))

def q(vals, frac):
    v = sorted(vals); return v[int(len(v)*frac)]

derived = {"matched_groups": len(matched), "n_strategies": len(common)}
for s in ("naive", "base", "oracle"):
    v = [S[s][k]["sharpe"] for k in common]
    tot = sum(S[s][k]["signals"] for k in common)
    derived[s] = {
        "pos": sum(1 for x in v if x > 0), "med": st.median(v),
        "q1": q(v, 0.25), "q3": q(v, 0.75), "min": min(v), "max": max(v),
        "win": sum(S[s][k]["win"]*S[s][k]["signals"] for k in common)/tot*100,
        "pnl": st.median([S[s][k]["pnl"] for k in common])*100,
        "signals": tot,
    }
derived["total_signals"] = sum(derived[s]["signals"] for s in ("naive","base","oracle"))
for a, b in (("base","naive"),("oracle","base"),("oracle","naive")):
    derived[f"{a}>{b}"] = sum(1 for k in common if S[a][k]["sharpe"] > S[b][k]["sharpe"])
def sp(a, b):
    ra = {k:i for i,k in enumerate(sorted(common, key=lambda k: S[a][k]["sharpe"]))}
    rb = {k:i for i,k in enumerate(sorted(common, key=lambda k: S[b][k]["sharpe"]))}
    n = len(common)
    return 1 - 6*sum((ra[k]-rb[k])**2 for k in common)/(n*(n*n-1))
derived["rho_oracle_base"] = sp("oracle","base")
derived["rho_base_naive"] = sp("base","naive")

print("=== RE-DERIVED FROM DATABASE ===")
print(f"matched groups {derived['matched_groups']}   strategies {derived['n_strategies']}")
for s in ("naive","base","oracle"):
    d = derived[s]
    print(f"  {s:7s} {d['pos']:2d}/{derived['n_strategies']} pos  med {d['med']:+7.3f}  q1 {d['q1']:+6.2f}  q3 {d['q3']:+6.2f}"
          f"  min {d['min']:+7.2f} max {d['max']:+7.2f}  win {d['win']:5.2f}%  pnl {d['pnl']:+.4f}%  {d['signals']:,}")
print(f"  base>naive {derived['base>naive']}  oracle>base {derived['oracle>base']}  oracle>naive {derived['oracle>naive']}")
print(f"  rho oracle~base {derived['rho_oracle_base']:+.3f}   rho base~naive {derived['rho_base_naive']:+.3f}")
print(f"  total signals {derived['total_signals']:,}")

# ---- check against the published page -------------------------------------
page = PAGE.read_text()
txt = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)))
# everything from the revision note onward legitimately quotes superseded figures
body = txt[:txt.find("Revision note")]

checks = [
    (f"{derived['matched_groups']} asset groups", None),
    (f"{derived['matched_groups']} groups present in all three sweeps", None),
    (f"{derived['n_strategies']} distinct strategies", None),
    (f"{derived['naive']['pos']} of {derived['n_strategies']} strategies", None),
    (f"{derived['n_strategies']-derived['oracle']['pos']} of {derived['n_strategies']} strategies remain unprofitable", None),
    (f"{derived['base']['pos']} of {derived['n_strategies']} strategies", None),
    (f"{derived['base>naive']} of {derived['n_strategies']}", None),
    (f"{derived['oracle>naive']} of {derived['n_strategies']} strategies", None),
    (f"{derived['total_signals']:,} signals", None),
]
fails = []
for needle, _ in checks:
    if needle not in body:
        fails.append(needle)

print("\n=== PAGE CROSS-CHECK ===")
if fails:
    print("MISSING from page body:")
    for f in fails: print("   ", repr(f))
else:
    print("all structural counts found in page body")

# numeric spot-checks on the summary table cells
for s in ("naive","base","oracle"):
    d = derived[s]
    for label, val, fmt in (("median", d["med"], "%+.3f"), ("q1", d["q1"], "%+.2f"),
                            ("q3", d["q3"], "%+.2f"), ("win", d["win"], "%.2f%%"),
                            ("pnl", d["pnl"], "%+.3f%%")):   # page renders pnl at 3dp, not 4
        needle = (fmt % val)
        if needle not in txt:
            print(f"   table cell not found: {s}.{label} = {needle}")

sys.exit(1 if fails else 0)
