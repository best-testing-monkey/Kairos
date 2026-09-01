#!/usr/bin/env python3
"""v3: market/instrument segmentation with the trend_following alias cluster
collapsed.

Eight strategy names wrap TrendFollowingStrategy() and gate on context keys the
pipeline never supplies (cds_spread_change, dark_pool_sentiment, ...), so every
gate defaults to a no-op pass-through and all eight are the same strategy.
Left uncollapsed they contribute seven redundant copies to every median.
"""
import sqlite3, re, json, pathlib, statistics as st
from collections import defaultdict

DB = "/media/baz/MonkeyWorks/PycharmProjects/Kairos/data/pipeline_results.db"
HERE = pathlib.Path(__file__).parent
MIN_SIGNALS, MIN_GROUPS = 3, 5
# One walk-forward exit rule for all three stages landed 2026-08-29; the re-sweep
# that followed starts on this date. Earlier rows were scored one bar ahead and
# force-closed at that bar's close -- a different ruler, so they are excluded.
EXIT_RULE_CUTOFF = "2026-08-30"

ALIAS = {"onchain_flow_filter", "insider_cluster", "gaussian_process", "fractal_dimension",
         "dark_pool_filter", "cot_positioning_filter", "cds_spread_filter"}  # drop; keep trend_following

conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
SUFFIX = re.compile(r"\.([A-Z]+)$")
VENUE = {"US": "United States", "L": "London", "HK": "Hong Kong", "DE": "Germany",
         "ST": "Stockholm", "AX": "Australia", "SW": "Switzerland", "PA": "Paris",
         "T": "Tokyo", "TO": "Toronto", "MI": "Milan", "AS": "Amsterdam", "MX": "Mexico"}

def gvenue(a):
    vs = {(SUFFIX.search(s).group(1) if SUFFIX.search(s) else "US") for s in a.split(",")}
    return vs.pop() if len(vs) == 1 else "mixed"

def weighted(rows, keyfn):
    agg = defaultdict(lambda: defaultdict(lambda: {"g":0,"s":0,"ws":0.0,"ww":0.0,"wp":0.0}))
    for r in rows:
        if r["strategy_name"] in ALIAS:      # collapse
            continue
        b = keyfn(r)
        if b is None: continue
        d = agg[b][r["strategy_name"]]; n = r["signal_count"]
        d["g"] += 1; d["s"] += n
        d["ws"] += r["sharpe"]*n; d["ww"] += r["win_rate"]*n; d["wp"] += r["avg_pnl_per_trade"]*n
    return {b: {k: {"groups":v["g"],"signals":v["s"],"sharpe":v["ws"]/v["s"],
                    "win":v["ww"]/v["s"],"pnl":v["wp"]/v["s"]}
                for k,v in stt.items() if v["g"] >= MIN_GROUPS}
            for b, stt in agg.items()}

def spearman(a,b,keys):
    keys=list(keys)
    if len(keys)<3: return None
    ra={k:i for i,k in enumerate(sorted(keys,key=lambda k:a[k]["sharpe"]))}
    rb={k:i for i,k in enumerate(sorted(keys,key=lambda k:b[k]["sharpe"]))}
    n=len(keys)
    return 1-6*sum((ra[k]-rb[k])**2 for k in keys)/(n*(n*n-1))

CLASSES=("equity","crypto","fx_commodity")
STAGES=["oracle","naive","base"]
report={"alias_collapsed":sorted(ALIAS),"by_class":{},"by_venue":{},"rank":{},
        "divergence":{},"summary":{},"venue_summary":{},
        "matched_cells":{}}

# Raw rows per stage, kept so the matched-cell comparison below can be derived
# instead of transcribed into build_market_report.py's prose by hand.
RAW={}

for stage in STAGES:
    rows=conn.execute("""SELECT s.strategy_name,s.sharpe,s.signal_count,s.win_rate,s.avg_pnl_per_trade,
                                s.assets,s.asset_class
                         FROM strategy_class_stats s JOIN runs r ON r.run_id=s.run_id
                         WHERE s.stage=? AND s.asset_class!='mixed' AND s.signal_count>=?
                           AND r.timestamp>=?""",
                      (stage,MIN_SIGNALS,EXIT_RULE_CUTOFF)).fetchall()
    ngroups=conn.execute("""SELECT COUNT(DISTINCT s.assets) FROM strategy_class_stats s
                            JOIN runs r ON r.run_id=s.run_id
                            WHERE s.stage=? AND r.timestamp>=?""",(stage,EXIT_RULE_CUTOFF)).fetchone()[0]
    report.setdefault("coverage",{})[stage]=ngroups
    RAW[stage]=rows
    byc=weighted(rows, lambda r:(r["asset_class"] if r["asset_class"] in CLASSES else None))
    have=[c for c in CLASSES if byc.get(c)]
    common=sorted(set.intersection(*[set(byc[c]) for c in have])) if have else []
    report["by_class"][stage]={c:byc[c] for c in have}

    print(f"\n{'='*74}\n{stage.upper()}   (common strategy set n={len(common)}, aliases collapsed)\n{'='*74}")
    report["summary"][stage]={"common_n":len(common),"classes":{}}
    for c in have:
        d=byc[c]; sub=sorted(d[k]["sharpe"] for k in common)
        tot=sum(d[k]["signals"] for k in common)
        win=sum(d[k]["win"]*d[k]["signals"] for k in common)/tot
        gps=sum(x['groups'] for x in d.values())//max(len(d),1)
        report["summary"][stage]["classes"][c]={
            "median":st.median(sub),"pos":sum(1 for v in sub if v>0),"of":len(sub),
            "win":win*100,"signals":tot,"groups_per_strat":gps}
        print(f"  {c:13s} median {st.median(sub):+8.3f}  {sum(1 for v in sub if v>0):2d}/{len(sub)} pos"
              f"  win {win*100:5.2f}%  {tot:>9,d} sig  ({len(d)} strat, {gps} grp/strat)")
    rk={}
    for i in range(len(have)):
        for j in range(i+1,len(have)):
            a,b=have[i],have[j]; rk[f"{a}|{b}"]=spearman(byc[a],byc[b],common)
    report["rank"][stage]=rk
    if rk:
        print("  strategy-ranking agreement across classes (Spearman, same common set):")
        for k,v in rk.items():
            a,b=k.split("|"); print(f"    {a:13s} vs {b:13s}  rho = {v:+.3f}")
    if len(have)>=2 and common:
        div=sorted(common,key=lambda k:-(max(byc[c][k]["sharpe"] for c in have)-min(byc[c][k]["sharpe"] for c in have)))
        report["divergence"][stage]=[{ "name":k, **{c:byc[c][k]["sharpe"] for c in have}} for k in div]
        print("  most class-dependent strategies (spread across classes):")
        for k in div[:6]:
            print(f"    {k:28s} " + "  ".join(f"{c[:6]:>6s} {byc[c][k]['sharpe']:+8.2f}" for c in have))

    # venue, equities only
    eqrows=[r for r in rows if r["asset_class"]=="equity"]
    byv=weighted(eqrows, lambda r:(lambda v: v if v!="mixed" else None)(gvenue(r["assets"])))
    byv={v:d for v,d in byv.items() if len(d)>=10}
    report["by_venue"][stage]=byv
    if byv:
        print("  equity by listing venue:")
        report["venue_summary"][stage]=[]
        for v,d in sorted(byv.items(),key=lambda kv:-sum(x["signals"] for x in kv[1].values())):
            sub=sorted(x["sharpe"] for x in d.values()); tot=sum(x["signals"] for x in d.values())
            shared=sorted(set(byv.get("US",{}))&set(d))
            rho=spearman(byv["US"],d,shared) if "US" in byv and v!="US" and len(shared)>=8 else None
            rtxt=f"  rho_vs_US {rho:+.3f}" if rho is not None else ""
            report["venue_summary"][stage].append(
                {"venue":VENUE.get(v,v),"strat":len(d),"median":st.median(sub),"signals":tot,"rho":rho})
            print(f"    {VENUE.get(v,v):16s} {len(d):3d} strat  median {st.median(sub):+7.3f}"
                  f"  {tot:>9,d} sig{rtxt}")


# ---------------------------------------------------------------------------
# Levels are not comparable across stages on their own, because the stages cover
# different group samples. These are the (group, strategy) cells where all three
# stages actually ran, which is the only like-for-like comparison available.
print(f"\n{'='*74}\nMATCHED (group, strategy) CELLS -- all three stages\n{'='*74}")
cells={stage:{(r["assets"],r["strategy_name"],r["asset_class"]):r["sharpe"]
              for r in RAW[stage] if r["strategy_name"] not in ALIAS}
       for stage in STAGES}
shared=set.intersection(*(set(c) for c in cells.values()))
for c in CLASSES:
    keys=[k for k in shared if k[2]==c]
    if len(keys)<MIN_GROUPS: continue
    report["matched_cells"][c]={"n":len(keys),
        **{stage:st.median([cells[stage][k] for k in keys]) for stage in STAGES}}
    m=report["matched_cells"][c]
    print(f"  {c:13s} n={m['n']:5d}  " + "  ".join(f"{s} {m[s]:+7.2f}" for s in STAGES))

json.dump(report,open(HERE/"market_analysis3.json","w"),indent=0)
print(f"\nwrote {HERE/'market_analysis3.json'}")
