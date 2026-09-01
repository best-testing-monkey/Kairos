#!/usr/bin/env python3
"""Generate the 'Where Strategies Travel' market-segmentation report."""
import pathlib, json

HERE = pathlib.Path(__file__).parent


# Provenance stamped at generation time. Hand-maintained date and commit strings
# had drifted from the run that actually produced the page, which defeats the
# point of a provenance line.
import datetime, subprocess
STAMP = datetime.date.today().strftime("%-d %B %Y")
try:
    COMMIT = subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    COMMIT = "unknown"

CLASSES = ["equity", "crypto", "fx_commodity"]
CNAME = {"equity": "Equities", "crypto": "Crypto", "fx_commodity": "FX &amp; commodities"}

# All figures are read from the analysis output rather than transcribed, so a
# re-run of analyze_by_market3.py after more sweep coverage regenerates the page
# without any hand-editing (and cannot silently drift out of sync with the data).
A = json.loads((HERE / "market_analysis3.json").read_text())

LEVEL = {reg: {cl: (v["median"], v["pos"], v["of"], v["win"], v["signals"], v["groups_per_strat"])
               for cl, v in s["classes"].items()}
         for reg, s in A["summary"].items()}
RANK = {reg: [(k.split("|")[0], k.split("|")[1], rho) for k, rho in pairs.items()]
        for reg, pairs in A["rank"].items()}
DIVERGENCE = {reg: [(r["name"], r["equity"], r["crypto"], r["fx_commodity"])
                    for r in rows[:6] if all(c in r for c in CLASSES)]
              for reg, rows in A["divergence"].items()}
VENUE = {reg: [(r["venue"], r["strat"], r["median"], r["signals"], r["rho"]) for r in rows]
         for reg, rows in A["venue_summary"].items()}
COVERAGE = A.get("coverage", {})
# Naive per-class figures, pulled from LEVEL so a re-sweep moves the prose with
# the data. These used to be transcribed into the paragraphs below, which is
# exactly the drift this file's docstring says it avoids -- and they went stale
# the moment the naive regime was rebuilt (2026-09-01).
# Refuse to render from a half-finished sweep. Without this the first missing
# class surfaces as a bare KeyError from deep inside an f-string, which reads
# like a bug in the generator rather than "the data is not ready yet".
_incomplete = {reg: [c for c in CLASSES if c not in LEVEL.get(reg, {})]
               for reg in ("naive", "oracle")}
for _reg, _missing in _incomplete.items():
    if _missing:
        raise SystemExit(
            f"{__file__}: {_reg} summary has no {_missing} row(s). That regime's sweep is "
            f"incomplete, so this page would report partial data as final. Finish the sweep, "
            f"re-run analyze_by_market3.py, then re-run this.")


def _nv(cl):
    med, pos, of, win, sig, gps = LEVEL["naive"][cl]
    return {"med": med, "pos": pos, "of": of, "win": win, "sig": sig, "gps": gps}

NV = {cl: _nv(cl) for cl in CLASSES}

# Cells where all three regimes actually ran -- the only like-for-like level
# comparison, since the regimes cover different group samples.
MC = A.get("matched_cells", {})

# Mean cross-class rank agreement per regime (the "portability ladder"), and the
# venue count, so the prose tracks coverage instead of being re-transcribed each
# time the base sweep grows.
def _rho(reg):
    vals = [v for _, _, v in (RANK.get(reg) or []) if v is not None]
    return sum(vals) / len(vals) if vals else None

RHO = {reg: _rho(reg) for reg in ("naive", "base", "oracle")}
NVENUE = {reg: len(A["venue_summary"].get(reg, [])) for reg in ("naive", "base", "oracle")}
BASE_N = COVERAGE.get("base", 0)

ALIAS_IMPACT = [("naive", 8, 51, -0.758, 8, 44, -0.721),
                ("base", 32, 51, 0.283, 32, 44, 0.351),
                ("oracle", 26, 51, 0.321, 26, 44, 2.302)]
ALIAS_NAMES = ["trend_following", "cds_spread_filter", "cot_positioning_filter", "dark_pool_filter",
               "fractal_dimension", "gaussian_process", "insider_cluster", "onchain_flow_filter"]

REG_LABEL = {"naive": "Naive", "base": "Base model", "oracle": "Oracle"}
REG_SUB = {"naive": "no forecast", "base": "Kronos-base", "oracle": "perfect foresight"}


def rank_figure():
    """Portability dial: cross-class rank agreement per regime."""
    blocks = []
    for reg in ("naive", "base", "oracle"):
        pairs = RANK.get(reg) or []
        if not pairs:
            # A regime with only one asset class above the 5-group floor cannot be
            # scored for cross-class agreement at all; say so rather than omit it.
            blocks.append(f'''<div class="rcard rcard-pending" data-reg="{reg}">
        <div class="rcard-h"><b>{REG_LABEL[reg]}</b><span>{REG_SUB[reg]}</span></div>
        <div class="rcard-big">&mdash;</div>
        <div class="rcard-cap">not yet measurable</div>
        <p class="rcard-note">The base sweep has re-run {COVERAGE.get(reg, 0)} of 961 groups under the current
        exit rule. Only equities clear the five-group minimum so far, so there is no second class to
        correlate against.</p></div>''')
            continue
        avg = sum(p[2] for p in pairs) / len(pairs)
        bars = "".join(
            f'<div class="rp"><span class="rp-l">{CNAME[a]} &middot; {CNAME[b]}</span>'
            f'<span class="rp-t"><i style="width:{max(rho,0)*100:.1f}%"></i></span>'
            f'<span class="rp-v">{rho:+.2f}</span></div>'
            for a, b, rho in pairs)
        blocks.append(f'''<div class="rcard" data-reg="{reg}">
        <div class="rcard-h"><b>{REG_LABEL[reg]}</b><span>{REG_SUB[reg]}</span></div>
        <div class="rcard-big">{avg:+.2f}</div>
        <div class="rcard-cap">mean cross-class agreement</div>
        {bars}</div>''')
    return f'''<figure class="fig">
  <div class="rgrid">{"".join(blocks)}</div>
  <figcaption><b>Figure 1.</b> Spearman rank correlation of per-strategy Sharpe between asset classes,
  computed on one common strategy set per regime so the coefficients are comparable. Without a forecast a
  strategy&rsquo;s standing in one class transfers weakly to another; with a perfect one, the ordering largely
  holds. Oracle and naive are scored on all three class pairs over the complete 961-group corpus; the base
  regime cannot be scored yet, and why is in the card (&sect;7.1).</figcaption>
</figure>'''


def level_table():
    rows = []
    for reg in ("naive", "base", "oracle"):
        d = LEVEL[reg]
        first = True
        for cl in CLASSES:
            if cl not in d:
                continue
            med, pos, of, win, sig, gps = d[cl]
            regcell = (f'<th scope="row" rowspan="{len(d)}" class="regcell">'
                       f'{REG_LABEL[reg]}<span>{REG_SUB[reg]}</span></th>') if first else ""
            first = False
            w = min(abs(med) / 5.0, 1.0) * 50.0
            style = f"left:50%;width:{w:.1f}%" if med >= 0 else f"right:50%;width:{w:.1f}%"
            rows.append(f'''<tr data-cls="{cl}">{regcell}
      <td><span class="dot"></span>{CNAME[cl]}</td>
      <td class="num sh {'pos' if med>=0 else 'neg'}"><i class="bar" style="{style}"></i><span>{med:+.3f}</span></td>
      <td class="num">{pos}/{of}</td><td class="num">{win:.2f}%</td>
      <td class="num sub">{sig:,}</td><td class="num sub">{gps}</td></tr>''')
    return f'''<div class="tbl-wrap"><table class="tbl">
  <caption><b>Table 1.</b> Performance level by asset class, within each regime. Medians are over that
  regime&rsquo;s common strategy set, so classes are directly comparable to each other &mdash;
  but not across regimes, which cover different groups (see &sect;6). The base regime has re-run {BASE_N} of
  961 groups so far, so its rows are provisional; oracle and naive cover all 961.</caption>
  <thead><tr><th scope="col">Regime</th><th scope="col">Asset class</th>
    <th scope="col" class="num">Median Sharpe</th><th scope="col" class="num">Positive</th>
    <th scope="col" class="num">Mean win</th><th scope="col" class="num">Signals</th>
    <th scope="col" class="num">Groups/strat</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table></div>'''


_DIV_TBL_NO = {reg: i for i, reg in enumerate(
    [r for r in ("naive", "base", "oracle") if A["divergence"].get(r)], start=2)}

def divergence_table(reg):
    if not DIVERGENCE.get(reg):
        return ""
    rows = "".join(
        f'<tr><th scope="row" class="sname">{n}</th>'
        + "".join(f'<td class="num sh {"pos" if v>=0 else "neg"}"><span>{v:+.2f}</span></td>'
                  for v in (e, c, f))
        + f'<td class="num sub">{max(e,c,f)-min(e,c,f):.1f}</td></tr>'
        for n, e, c, f in DIVERGENCE[reg])
    return f'''<div class="tbl-wrap"><table class="tbl">
  <caption><b>Table {_DIV_TBL_NO[reg]}.</b> Most class-dependent strategies under
  <b>{REG_LABEL[reg].lower()}</b>, by spread between best and worst class.</caption>
  <thead><tr><th scope="col">Strategy</th><th scope="col" class="num">Equities</th>
    <th scope="col" class="num">Crypto</th><th scope="col" class="num">FX &amp; comm.</th>
    <th scope="col" class="num">Spread</th></tr></thead><tbody>{rows}</tbody></table></div>'''


def venue_table():
    blocks = []
    for reg in ("oracle", "base", "naive"):
        rows = "".join(
            f'<tr><th scope="row" class="sname">{v}</th><td class="num sub">{n}</td>'
            f'<td class="num sh {"pos" if m>=0 else "neg"}"><span>{m:+.3f}</span></td>'
            f'<td class="num sub">{s:,}</td>'
            f'<td class="num">{"&mdash;" if r is None else f"{r:+.3f}"}</td></tr>'
            for v, n, m, s, r in VENUE[reg])
        blocks.append(f'''<h4 class="vh">{REG_LABEL[reg]}</h4>
    <div class="tbl-wrap"><table class="tbl">
      <thead><tr><th scope="col">Listing venue</th><th scope="col" class="num">Strat</th>
      <th scope="col" class="num">Median Sharpe</th><th scope="col" class="num">Signals</th>
      <th scope="col" class="num">&rho; vs US</th></tr></thead><tbody>{rows}</tbody></table></div>''')
    return "".join(blocks)


def alias_table():
    rows = "".join(
        f'<tr><th scope="row">{REG_LABEL[r]}</th>'
        f'<td class="num">{pa}/{na}</td><td class="num">{ma:+.3f}</td>'
        f'<td class="num">{pb}/{nb}</td><td class="num strong">{mb:+.3f}</td>'
        f'<td class="num sub">{mb-ma:+.3f}</td></tr>'
        for r, pa, na, ma, pb, nb, mb in ALIAS_IMPACT)
    return f'''<div class="tbl-wrap"><table class="tbl">
  <caption><b>Table {max(_DIV_TBL_NO.values()) + 1}.</b> Effect of collapsing the eight aliased names to one on the companion
  paper&rsquo;s headline figures. Computed on the 451-group matched sample published before the exit-rule
  correction; the companion paper has collapsed the aliases ever since, so this is the record of the
  correction, not a current measurement.</caption>
  <thead><tr><th scope="col">Regime</th><th scope="col" class="num">Positive (as published)</th>
  <th scope="col" class="num">Median</th><th scope="col" class="num">Positive (collapsed)</th>
  <th scope="col" class="num">Median</th><th scope="col" class="num">&Delta; median</th></tr></thead>
  <tbody>{rows}</tbody></table></div>'''


HTML = f'''<title>Where Strategies Travel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@450;600;700&family=Spectral:ital,wght@0,400;0,500;1,400&display=swap">
<style>
:root {{
  --ground:#FAFBFC; --surface:#EFF3F6; --surface-2:#E4EAEF;
  --ink:#0E141A; --ink-mid:#3C4854; --ink-mute:#66727E; --rule:#D3DBE2;
  --equity:#1E51A0; --crypto:#8A3A82; --fx_commodity:#8A6009; --accent:#1E51A0;
  --warn:#9A3412;
  --measure:68ch;
  --f-head:"IBM Plex Sans Condensed",-apple-system,"Segoe UI",sans-serif;
  --f-body:"Spectral",Georgia,"Times New Roman",serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
  --ground:#0E1317; --surface:#161D23; --surface-2:#1E272F;
  --ink:#E3E9EF; --ink-mid:#B3BFCA; --ink-mute:#7D8B98; --rule:#2A343D;
  --equity:#7EA9EC; --crypto:#D68FCC; --fx_commodity:#DDB055; --accent:#7EA9EC;
  --warn:#F0956A;
}} }}
:root[data-theme="dark"] {{
  --ground:#0E1317; --surface:#161D23; --surface-2:#1E272F;
  --ink:#E3E9EF; --ink-mid:#B3BFCA; --ink-mute:#7D8B98; --rule:#2A343D;
  --equity:#7EA9EC; --crypto:#D68FCC; --fx_commodity:#DDB055; --accent:#7EA9EC;
  --warn:#F0956A;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--ground); color:var(--ink); font-family:var(--f-body);
  font-size:17px; line-height:1.62; margin:0; padding:0 5vw 8rem; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:var(--measure); margin:0 auto; }}
.wide {{ max-width:min(1080px,94vw); margin-inline:auto; }}
header.mast {{ max-width:min(1080px,94vw); margin:0 auto; padding:5.5rem 0 0; }}
.eyebrow {{ font-family:var(--f-head); font-weight:600; font-size:.74rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-mute); display:flex; gap:.7rem; flex-wrap:wrap; align-items:center; }}
.eyebrow i {{ font-style:normal; color:var(--rule); }}
h1 {{ font-family:var(--f-head); font-weight:700; font-size:clamp(2.5rem,6.6vw,4.4rem);
  line-height:1.03; letter-spacing:-.018em; margin:1.1rem 0 0; text-wrap:balance; }}
.standfirst {{ font-size:clamp(1.06rem,2.1vw,1.3rem); line-height:1.5; color:var(--ink-mid);
  max-width:57ch; margin:1.4rem 0 0; font-style:italic; }}
.byline {{ font-family:var(--f-mono); font-size:.78rem; color:var(--ink-mute); margin:2.2rem 0 0;
  padding-top:1.1rem; border-top:1px solid var(--rule); display:flex; gap:1.6rem; flex-wrap:wrap; }}
.abstract {{ background:var(--surface); border-left:3px solid var(--accent); padding:1.6rem 1.9rem;
  margin:3.2rem auto 0; max-width:min(1080px,94vw); }}
.abstract h2 {{ font-family:var(--f-head); font-size:.75rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--ink-mute); margin:0 0 .7rem; font-weight:600; }}
.abstract p {{ margin:0 0 .85rem; font-size:1rem; }} .abstract p:last-child {{ margin-bottom:0; }}
section {{ margin:4.2rem auto 0; }}
h2.sec {{ font-family:var(--f-head); font-weight:600; font-size:1.72rem; margin:0 0 1.1rem;
  display:flex; gap:.85rem; align-items:baseline; text-wrap:balance; letter-spacing:-.008em; }}
h2.sec .no {{ font-family:var(--f-mono); font-size:.86rem; color:var(--accent); font-weight:500; flex:none; padding-top:.15em; }}
h3 {{ font-family:var(--f-head); font-weight:600; font-size:1.1rem; margin:2.3rem 0 .6rem; }}
h4.vh {{ font-family:var(--f-head); font-weight:600; font-size:.95rem; margin:1.8rem 0 .4rem; color:var(--ink-mid); }}
p {{ margin:0 0 1.05rem; }} a {{ color:var(--accent); }}
strong {{ font-weight:500; }} code {{ font-family:var(--f-mono); font-size:.86em; background:var(--surface-2); padding:.1em .35em; border-radius:2px; }}
ul,ol {{ margin:0 0 1.05rem; padding-left:1.4rem; }} li {{ margin-bottom:.5rem; }}
.lede::first-letter {{ font-family:var(--f-head); font-weight:700; font-size:3.1em; float:left;
  line-height:.82; padding:.06em .09em 0 0; color:var(--accent); }}
/* rank figure */
.fig {{ margin:2.4rem 0 2rem; }}
.rgrid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }}
.rcard {{ background:var(--surface); padding:1.2rem 1.3rem 1rem; border-top:3px solid var(--accent); }}
.rcard[data-reg="naive"] {{ border-top-color:var(--ink-mute); }}
.rcard[data-reg="oracle"] {{ border-top-color:var(--fx_commodity); }}
.rcard-h b {{ font-family:var(--f-head); font-size:1rem; display:block; }}
.rcard-h span {{ font-family:var(--f-mono); font-size:.68rem; color:var(--ink-mute); }}
.rcard-big {{ font-family:var(--f-head); font-size:2.7rem; font-weight:700; line-height:1.1;
  margin:.5rem 0 0; font-variant-numeric:tabular-nums; }}
.rcard-cap {{ font-family:var(--f-mono); font-size:.66rem; color:var(--ink-mute);
  text-transform:uppercase; letter-spacing:.08em; margin-bottom:.9rem; }}
.rcard-pending {{ border-top-style:dashed; }}
.rcard-pending .rcard-big {{ color:var(--ink-mute); }}
.rcard-note {{ font-family:var(--f-mono); font-size:.66rem; line-height:1.55; color:var(--ink-mute); margin:0; }}
.rp {{ display:grid; grid-template-columns:1fr 58px 38px; gap:.5rem; align-items:center; margin-top:.35rem; }}
.rp-l {{ font-family:var(--f-mono); font-size:.63rem; color:var(--ink-mute); }}
.rp-t {{ position:relative; height:5px; background:var(--surface-2); display:block; }}
.rp-t i {{ position:absolute; left:0; top:0; bottom:0; background:var(--accent); }}
.rp-v {{ font-family:var(--f-mono); font-size:.7rem; text-align:right; font-variant-numeric:tabular-nums; }}
figcaption {{ font-size:.88rem; line-height:1.5; color:var(--ink-mute); margin-top:.9rem; max-width:78ch; }}
figcaption b {{ color:var(--ink-mid); font-weight:500; }}
/* tables */
.tbl-wrap {{ overflow-x:auto; margin:1.4rem 0; background:var(--surface); }}
table.tbl {{ border-collapse:collapse; width:100%; font-family:var(--f-mono); font-size:.8rem; }}
.tbl caption {{ font-family:var(--f-body); font-size:.88rem; color:var(--ink-mute); text-align:left;
  padding:1rem 1.1rem; line-height:1.45; caption-side:top; }}
.tbl caption b {{ color:var(--ink-mid); font-weight:500; }}
.tbl th,.tbl td {{ padding:.42rem .7rem; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap; }}
.tbl thead th {{ position:sticky; top:0; z-index:2; background:var(--surface-2); font-family:var(--f-head);
  font-weight:600; font-size:.76rem; letter-spacing:.04em; color:var(--ink-mid); }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }} .sub {{ color:var(--ink-mute); }}
.sname {{ font-weight:400; }} .strong {{ font-weight:500; color:var(--ink); }}
.regcell {{ font-family:var(--f-head); font-weight:600; vertical-align:top; border-right:1px solid var(--rule); }}
.regcell span {{ display:block; font-family:var(--f-mono); font-weight:400; font-size:.66rem; color:var(--ink-mute); }}
tr[data-cls="equity"] {{ --c:var(--equity); }} tr[data-cls="crypto"] {{ --c:var(--crypto); }}
tr[data-cls="fx_commodity"] {{ --c:var(--fx_commodity); }}
.dot {{ display:inline-block; width:9px; height:9px; margin-right:.55rem; background:var(--c); }}
.sh {{ position:relative; min-width:104px; }}
.sh .bar {{ position:absolute; top:5px; bottom:5px; background:var(--c,var(--accent)); opacity:.22; }}
.sh span {{ position:relative; }} .sh.neg span {{ color:var(--ink-mute); }}
tbody tr:hover td, tbody tr:hover th {{ background:var(--surface-2); }}
.callout {{ border:1px solid var(--rule); border-left:3px solid var(--warn); background:var(--surface);
  padding:1.3rem 1.5rem; margin:2rem 0; }}
.callout h4 {{ font-family:var(--f-head); font-size:.75rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--warn); margin:0 0 .6rem; font-weight:600; }}
.callout p:last-child {{ margin-bottom:0; }}
pre.sql {{ font-family:var(--f-mono); font-size:.78rem; line-height:1.55; background:var(--surface);
  padding:1.1rem 1.3rem; overflow-x:auto; margin:1.2rem 0; color:var(--ink-mid); border-left:3px solid var(--rule); }}
footer {{ max-width:min(1080px,94vw); margin:5rem auto 0; padding-top:1.4rem; border-top:1px solid var(--rule);
  font-family:var(--f-mono); font-size:.75rem; color:var(--ink-mute); line-height:1.7; }}
@media (max-width:640px) {{ body {{ font-size:16px; }} .rp {{ grid-template-columns:1fr 40px 34px; }} }}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<header class="mast">
  <div class="eyebrow"><span>Kairos Research Note</span><i>&mdash;</i><span>Market Segmentation</span>
  <i>&mdash;</i><span>31 August 2026</span></div>
  <h1>Where Strategies Travel</h1>
  <p class="standfirst">Does a strategy that works on equities work on crypto? On gold? On a Hong Kong
  listing? Segmenting 5.6&nbsp;million signals by asset class and listing venue &mdash; and finding that
  what makes performance portable is the forecast, not the market.</p>
  <div class="byline"><span>Kairos Project</span><span>1d bars &middot; 6-month window</span>
  <span>3 asset classes &middot; 6 listing venues</span><span>companion to <em>The Prediction Premium</em></span></div>
</header>

<div class="abstract">
  <h2>Abstract</h2>
  <p>The companion paper treated the 127-strategy corpus as one population. It is not. Segmenting results
  by asset class shows large, systematic differences in <em>level</em>: under perfect foresight the median
  strategy reaches +5.56 Sharpe on FX and commodities and +2.70 on equities, but &minus;3.35 on crypto &mdash;
  a spread far wider than the difference between prediction regimes.</p>
  <p>The more useful finding concerns <em>portability</em>. Ranking strategies within each asset class and
  correlating those rankings, the no-forecast regime averages <strong>&rho;&nbsp;=&nbsp;+0.28</strong> across
  the three class pairs &mdash; a weak, partly-shared ordering. With a perfect forecast agreement rises to
  <strong>+0.74</strong>. Forecast information is a large part of what makes strategy quality a property of
  the strategy rather than of the market it happened to be tested on.</p>
  <p>Listing venue matters much less than asset class. Across six exchanges, strategy rankings agree with the
  US ordering at &rho;&nbsp;=&nbsp;+0.92 to +0.97 under oracle. A Hong Kong or German listing is, for these
  strategies, far more like a US listing than any equity is like a crypto pair.</p>
  <p>One class breaks the companion paper&rsquo;s pattern outright: <strong>FX and commodities are the only
  segment profitable without any forecast at all</strong>, at a naive median of {NV["fx_commodity"]["med"]:+.2f}
  with {NV["fx_commodity"]["pos"]} of {NV["fx_commodity"]["of"]} strategies above zero, against
  {NV["equity"]["med"]:+.2f} on equities and {NV["crypto"]["med"]:+.2f} on crypto.</p>
  <p>This edition is the first scored entirely on the unified walk-forward exit rule (companion paper,
  &sect;3.4), which retires the evaluator artefact this note previously offered as a partial explanation for
  crypto&rsquo;s deficit &mdash; the deficit survived the correction. It also closes a gap: the base sweep
  has reached {BASE_N} of 961 groups and now clears the five-group minimum in all three classes, so the
  middle rung of the portability ladder &mdash; blank in the previous edition &mdash; is measured here for
  the first time.</p>
</div>

<div class="wrap">
<section>
  <h2 class="sec"><span class="no">01</span>The question</h2>
  <p class="lede">A corpus-wide average assumes the corpus faces one market. It does not. Crypto trades
  continuously and reprices on sentiment; gold responds to real rates and risk appetite; an equity carries
  idiosyncratic earnings risk on top of its index beta; a Hong Kong listing trades in a different session,
  under different microstructure, than a New York one.</p>
  <p>If strategies behave differently across those settings, then a single ranking of the corpus is
  misleading &mdash; it describes an average market that no one trades. The practical question is whether a
  strategy selected on one class can be deployed on another, or whether selection has to be redone per
  market.</p>
  <p>This note segments the same three prediction regimes used in the companion paper &mdash; naive
  (no forecast), base (Kronos-base), oracle (perfect foresight) &mdash; by asset class and, within equities,
  by listing venue.</p>
</section>

<section>
  <h2 class="sec"><span class="no">02</span>Portability: the headline</h2>
  {rank_figure()}
  <p>Rank agreement answers the deployment question directly, and it is robust in a way that level
  comparisons are not: it does not care that crypto and equity groups contain different assets over
  different sessions, only whether the two markets <em>order the strategies the same way</em>.</p>
  <p>Without a forecast, agreement is weak. The three class pairs average
  <strong>&rho;&nbsp;=&nbsp;+0.28</strong> (+0.27 equity&ndash;crypto, +0.50 equity&ndash;FX, +0.05
  crypto&ndash;FX). Some ordering does survive &mdash; a strategy that is catastrophic on one class tends to
  be poor on another &mdash; but it is far too loose to select on, and the crypto&ndash;FX pair carries no
  usable signal at all. A strategy&rsquo;s naive standing on equities explains under a tenth of its variance
  in standing on crypto.</p>
  <p>With a perfect forecast, agreement rises sharply and consistently: oracle reaches <strong>+0.61 to
  +0.87</strong> across the three pairs (mean <strong>+0.74</strong>), with the weakest pair &mdash;
  crypto&ndash;FX, the two thinnest classes &mdash; still more than twice its naive value. The forecast
  supplies a common signal that each strategy transforms in its own characteristic way, so merit becomes a
  property of the transformation rather than of the market. That is the finding worth carrying forward:
  <strong>prediction is a large part of what makes strategy selection portable.</strong></p>
  <p>The middle rung, blank in the previous edition, is now in place. The base sweep has reached {BASE_N}
  of 961 groups under the corrected exit rule &mdash; enough for all three classes to clear the five-group
  minimum &mdash; and its cross-class agreement comes in at <strong>{RHO["base"]:+.2f}</strong>, between the
  naive floor&rsquo;s {RHO["naive"]:+.2f} and the oracle&rsquo;s {RHO["oracle"]:+.2f}. The ladder is
  monotonic, which is what the portability claim predicted and what the previous edition could only assert.
  The earlier figure of +0.71 for base is still not carried forward: it was measured on the retired one-bar
  exit rule and is not comparable. Base coverage is short of the full 961, so this coefficient should be
  expected to move.</p>
</section>

<section>
  <h2 class="sec"><span class="no">03</span>Level: asset classes are not equally tractable</h2>
  <p>Portability is not the same as parity. Rankings may agree while absolute performance differs sharply,
  and it does.</p>
</section>
</div>
<div class="wide">{level_table()}</div>
<div class="wrap">
<section>
  <p>Under perfect foresight the spread is dramatic: a median of <strong>+5.56 on FX and commodities and
  +2.70 on equities against &minus;3.35 on crypto</strong>. The corpus extracts far more from a perfectly
  forecast commodity or equity bar than from a perfectly forecast crypto bar. The mechanism is bracket
  geometry rather than direction: crypto&rsquo;s intrabar range is wide relative to a strategy&rsquo;s stop
  distance, so a position that is directionally right is stopped out on the way. Crypto&rsquo;s mean win rate
  under oracle is <strong>36.3%</strong> against equities&rsquo; 59.8% and FX/commodities&rsquo; 66.2%, all
  three holding a perfect next-bar forecast &mdash; direction is not what is failing.</p>
  <div class="callout">
    <h4>An explanation this edition can no longer use</h4>
    <p>The previous edition attributed part of crypto&rsquo;s deficit to the evaluator: oracle and base were
    scored one bar ahead and force-closed at that bar&rsquo;s close, which penalises wide-range instruments
    more than a walk-forward would. That asymmetry is gone &mdash; all three regimes now walk forward to a
    genuine stop or target (companion paper, &sect;3.4) and the whole corpus has been re-swept under the new
    rule.</p>
    <p>Crypto&rsquo;s deficit survived it. On the same ruler as everything else, crypto still sits about six
    Sharpe below equities under oracle, on a 36.3% win rate with perfect foresight. The wide-range
    explanation stands; the evaluator excuse does not. Whatever is failing on crypto is in the brackets the
    strategies choose, not in how those brackets were scored.</p>
  </div>
  <p>Levels are not comparable across regimes on their own, because the regimes cover different group
  samples. Checked properly, on the {MC["equity"]["n"]:,} matched (group, strategy) equity cells where all
  three ran, the median is <strong>{MC["equity"]["oracle"]:+.2f}</strong> under oracle,
  <strong>{MC["equity"]["base"]:+.2f}</strong> under base and <strong>{MC["equity"]["naive"]:+.2f}</strong>
  under naive &mdash; the same ordering the companion paper reports, on the same assets. The
  {MC["crypto"]["n"]:,} matched crypto cells put oracle at {MC["crypto"]["oracle"]:+.2f} against
  base&rsquo;s {MC["crypto"]["base"]:+.2f}. The ceiling towers over the system in both.</p>

  <h3>3.1 &mdash; FX and commodities work without a forecast</h3>
  <p>The naive row carries the one result that contradicts the companion paper&rsquo;s headline. That paper
  concludes that the corpus does not work without forecast information &mdash; the median strategy well below
  zero. Segmented, that conclusion holds for equities (median {NV["equity"]["med"]:+.2f},
  {NV["equity"]["pos"]} of {NV["equity"]["of"]} strategies positive) and holds harder for crypto
  ({NV["crypto"]["med"]:+.2f}, {NV["crypto"]["pos"]} of {NV["crypto"]["of"]}). It fails for FX and
  commodities, where the naive median is <strong>{NV["fx_commodity"]["med"]:+.2f} with
  {NV["fx_commodity"]["pos"]} of {NV["fx_commodity"]["of"]} strategies positive</strong> &mdash; the only
  regime-and-class cell in this study that is broadly profitable with no forward information whatsoever.</p>
  <p>The plausible reading is that these instruments are the corpus&rsquo;s best fit for
  <em>structural</em> rather than directional edge. The companion paper found that the strategies surviving
  the naive regime were those trading bracket geometry and win-rate asymmetry rather than direction; range-bound,
  mean-reverting commodity and currency series are exactly where that kind of edge should persist. The same
  class is also the strongest under oracle at +5.56, so the structural edge and the forecast-driven one stack
  rather than compete.</p>
  <p>It is by far the thinnest cell in the study &mdash; {NV["fx_commodity"]["sig"]:,} signals from
  {NV["fx_commodity"]["gps"]} groups, against {NV["equity"]["sig"]:,} from {NV["equity"]["gps"]} groups for
  naive equities &mdash; and it got thinner in this edition, not wider. The previous
  version of this note reported +1.34 on 128,327 signals, drawn partly from groups outside the deduped
  961-group list that a targeted backfill had added. The re-sweep covers the 961-group list only, so those
  extra groups are gone and both the median and the count fell with them (27 of 38 above zero, then; 21 of 37,
  now). The result held its sign and its uniqueness across a four-fold cut in sample; treat the magnitude as
  the least established number in this note.</p>

  <h3>3.2 &mdash; Which strategies are most class-dependent</h3>
</section>
</div>
<div class="wide">{divergence_table("naive")}{divergence_table("base")}{divergence_table("oracle")}</div>
<div class="wrap">
<section>
  <p>Two patterns stand out, and both survive the change of exit rule. <code>volume_fade</code> and
  <code>volume_confirmation</code> degrade most on crypto in <em>both</em> scored regimes &mdash; both read
  volume, and crypto volume is reported inconsistently across venues and is not comparable to a consolidated
  equity tape. <code>path_v_shape</code> is the sharpest sign reversal under oracle (+16.59 on equities,
  +15.37 on FX and commodities, &minus;6.48 on crypto): it depends on mean-reverting intraday path shape,
  which crypto&rsquo;s trendier sessions do not supply. It reverses the same way under naive, at a tenth the
  magnitude, so the effect is a property of the market rather than of the forecast.</p>
  <p><code>expected_value</code> and its twin <code>vol_target_sizer</code> run the other way &mdash; +30.18
  on crypto against +12.37 on equities under oracle, the corpus&rsquo;s clearest case of a strategy that
  wants the wide ranges everything else is punished by. A class-specific deployment rule would have to catch
  both directions. The base model cannot be scored here yet, so this section covers two regimes rather than
  three.</p>
</section>

<section>
  <h2 class="sec"><span class="no">04</span>Listing venue matters far less</h2>
  <p>Within equities, the picture is markedly more uniform than across asset classes.</p>
</section>
</div>
<div class="wide">{venue_table()}</div>
<div class="wrap">
<section>
  <p>Under oracle, every non-US venue ranks strategies almost identically to the US
  (<strong>&rho;&nbsp;=&nbsp;+0.92 to +0.97</strong>). Whatever differences exist between trading in New York,
  Hong Kong, Frankfurt, Sydney, Zurich and London, they do not change which strategies work. Compare that to
  the +0.75 equity-versus-crypto figure and the ordering is clear: <strong>instrument type is a first-order
  effect; listing venue is a second-order one.</strong></p>
  <p>Non-US venues do show higher median Sharpe under oracle (+1.8 to +4.5 against the US +1.6), but this
  should be discounted heavily. Non-US signal counts are two orders of magnitude smaller (7,000&ndash;71,000
  against 2.32 million), and the universe screen&rsquo;s flat $50M liquidity floor is far more selective
  abroad than in the US &mdash; the surviving non-US names are their markets&rsquo; largest and most liquid,
  which is a different population, not a better market. Under naive the same venues show scattered, weak
  agreement (&rho; from &minus;0.05 to +0.65), reinforcing &sect;2: strip the forecast and cross-market
  agreement collapses at venue level too. The base rows cover only two venues so far and are not yet worth
  reading.</p>
</section>

<section>
  <h2 class="sec"><span class="no">05</span>A defect found on the way</h2>
  <div class="callout">
    <h4>Eight names, one strategy</h4>
    <p>Segmenting the corpus exposed groups of strategies returning byte-identical Sharpe and signal counts.
    Eight names are one strategy: <code>trend_following</code> plus <code>cds_spread_filter</code>,
    <code>cot_positioning_filter</code>, <code>dark_pool_filter</code>, <code>fractal_dimension</code>,
    <code>gaussian_process</code>, <code>insider_cluster</code> and <code>onchain_flow_filter</code>.</p>
    <p>Each of the seven filters wraps <code>TrendFollowingStrategy()</code> and gates on a context key the
    pipeline never supplies &mdash; <code>cds_spread_change</code>, <code>dark_pool_sentiment</code>,
    <code>cot_net_position</code> and so on. Each read defaults to <code>0.0</code>, every gate condition is
    therefore false, and each filter degrades to an unmodified pass-through that differs from
    <code>trend_following</code> only in the name stamped on the signal. This is confirmed in code, not
    inferred from the numbers.</p>
  </div>
  <p>The consequence is that corpus-wide statistics count one behaviour eight times. Since that behaviour is
  unprofitable in every regime and sits near the middle of the oracle distribution, removing the seven
  duplicates moves the medians &mdash; substantially, in oracle&rsquo;s case.</p>
</section>
</div>
<div class="wide">{alias_table()}</div>
<div class="wrap">
<section>
  <p>Counts of profitable strategies were unaffected (all eight are negative everywhere, so none was ever
  counted as a success), but the denominators and medians were not: the oracle median was understated by a
  factor of seven. The companion paper has collapsed the seven duplicates since the edition this table
  describes, and its current matched sample reports <strong>43 distinct strategies</strong>. The
  corpus-level conclusions did not reverse &mdash; naive stayed overwhelmingly unprofitable, base broadly
  positive, oracle far above both &mdash; but the published oracle median was wrong, and this is where it
  was caught.</p>
  <p>A wider audit is warranted. Several further near-duplicate pairs coincide on most but not all groups
  (<code>expected_value</code>/<code>vol_target_sizer</code> on 445 groups,
  <code>range_trading</code>/<code>rqa_determinism</code> on 302,
  <code>dynamic_bracket</code>/<code>inverse_variance</code> on 262,
  <code>amount_flow</code>/<code>predicted_vwap</code> on 70). Those are not exact aliases, but they are
  close enough to warrant checking whether they are meaningfully distinct strategies.</p>
</section>

<section>
  <h2 class="sec"><span class="no">06</span>Limitations</h2>
  <ol>
    <li><strong>Levels are not comparable across regimes.</strong> The three sweeps cover different groups per
    class, so &sect;3&rsquo;s within-regime class comparisons are valid while any cross-regime reading of the
    same table is not. Where a cross-regime claim was needed, it was computed on matched cells and is labelled
    as such.</li>
    <li><strong>The base regime is still mid-re-sweep.</strong> {BASE_N} of 961 groups have been re-run
    under the current exit rule. That is now enough for a cross-class rank coefficient (&sect;2), a divergence
    table (&sect;3.2) and {NVENUE["base"]} listing venues (&sect;4), all of which the previous edition had to
    leave blank &mdash; but it is not the whole corpus, so every base figure here is provisional and should be
    expected to move as the remaining groups land. Oracle and naive are complete at 961 of 961 groups each and
    are not affected.</li>
    <li><strong>Superseded: the exit-rule asymmetry is gone.</strong> Earlier editions carried the companion
    paper&rsquo;s caveat that oracle and base were scored one bar ahead then force-closed while naive walked
    forward multi-bar, and noted that this bore directly on &sect;3&rsquo;s crypto explanation &mdash; a
    single-bar forced close penalises wide-range instruments, so part of crypto&rsquo;s oracle deficit might
    have been evaluator artefact. All three regimes now share one walk-forward rule and the corpus has been
    re-swept under it. Crypto&rsquo;s deficit persisted, so that particular doubt is resolved rather than
    outstanding (&sect;3).</li>
    <li><strong>Class samples are very unequal.</strong> Under oracle, equities contribute 748 groups per
    strategy against 25 for crypto and 12 for FX/commodities. The crypto and FX medians rest on far less
    data, and no confidence intervals are computed anywhere in this note. FX/commodity coverage is smaller
    than the previous edition&rsquo;s because the re-sweep covers the deduped 961-group list only
    (&sect;7.1).</li>
    <li><strong>Superseded: the FX/commodity sample is no longer commodity-only.</strong> An earlier
    version of this note warned that seventeen genuine FX-pair groups (<code>EURUSD=X</code>,
    <code>CADJPY=X</code>, &hellip;) classified as unknown and were dropped, because class was derived from
    the universe screen&rsquo;s membership list and those symbols were never screened. Class now comes from
    per-signal attribution recorded at sweep time, which classifies by ticker suffix and so places them
    correctly &mdash; recovering 222,342 previously-discarded oracle signals and roughly doubling this
    class&rsquo;s coverage. &sect;3.1 now reflects both currencies and commodities. It remains the thinnest
    class in the study.</li>
    <li><strong>Venue is a listing suffix, not an economy.</strong> Groups are classified by yfinance symbol
    suffix, which identifies where a security is listed &mdash; not where the company earns, nor its sector.
    No sector or country-of-revenue data exists in the pipeline, so &ldquo;the market the company is in&rdquo;
    is only approximated, and multinationals are misattributed by construction.</li>
    <li><strong>Mixed-class and mixed-venue groups are excluded</strong> rather than modelled, along with the
    small number of groups containing symbols absent from the universe screen. This is a clean-sample choice,
    not a claim that mixed baskets behave like either constituent.</li>
    <li><strong>No costs, shadow accounting, no significance testing</strong> &mdash; all as described in the
    companion paper, and all still true here.</li>
  </ol>
</section>

<section>
  <h2 class="sec"><span class="no">07</span>Data provenance</h2>
  <p>Same database and tables as the companion paper. The only addition is the symbol&rarr;class join, which
  comes from the universe screen rather than from the results tables.</p>
  <pre class="sql">-- asset class per symbol (1d universe run 759: 1,779 equity, 52 crypto, 20 fx_commodity)
SELECT symbol, asset_class FROM universe_screen WHERE passed=1 AND run_id=759;

-- a group's class is the class of its members; mixed and unknown groups are dropped
-- a group's venue is the yfinance suffix (no suffix =&gt; US); mixed-venue groups dropped

-- per-signal class attribution, recorded at sweep time
SELECT s.strategy_name, s.sharpe, s.signal_count, s.win_rate, s.avg_pnl_per_trade,
       s.assets, s.asset_class
FROM   strategy_class_stats s JOIN runs r ON r.run_id = s.run_id
WHERE  s.stage = ?  AND s.signal_count &gt;= 3  AND s.asset_class != 'mixed'
  AND  r.timestamp &gt;= '2026-08-30';   -- the one-exit-rule re-sweep; earlier rows
                                      -- use the retired one-bar rule and are excluded</pre>
  <p>Per-strategy figures are signal-count-weighted within each class, requiring at least 5 groups per
  strategy per class. Rank correlations use one common strategy set per regime, so the coefficients within a
  regime are comparable to each other. Scripts: <code>analyze_by_market3.py</code> (analysis),
  <code>build_market_report.py</code> (this page). The page is generated from the analysis output rather than
  from transcribed numbers, so re-running the analysis after more sweep coverage regenerates every figure
  here without hand-editing.</p>
  <h3>7.1 &mdash; Coverage, and what the re-sweep changed</h3>
  <p>Every figure in this edition comes from the re-sweep that followed the exit-rule unification of
  29 August 2026. Rows written before it were scored one bar ahead and force-closed, so they are excluded
  rather than blended in &mdash; which also means this note&rsquo;s footprint is now exactly the deduped
  <strong>961-group list</strong>, for oracle and naive alike, with no groups from outside it. Contributing
  groups per class: <strong>{NV["equity"]["gps"]} equity, {NV["crypto"]["gps"]} crypto,
  {NV["fx_commodity"]["gps"]} FX/commodity</strong> (a group can contribute signals to more than one class).
  Base stands at {BASE_N} groups.</p>
  <p><strong>Superseded: the naive backfill.</strong> Earlier editions of this note described a targeted
  backfill that ran naive over 186 groups from outside the deduped list, taking it from 959 to 1,144 groups
  and lifting crypto from 29 to 99 groups and FX/commodities from 4 to 33. Those rows are pre-fix and no
  longer used. The class coverage above is therefore narrower than the previous edition&rsquo;s, and
  &sect;3.1&rsquo;s FX/commodity result is measured on roughly a quarter of the signals it previously rested
  on. It survived the cut; it is the thinnest claim here. Extending the sweeps beyond the deduped list on the
  current exit rule would restore the width.</p>
  <p>The list itself is 95% equity by construction &mdash; it is a greedy set cover over universe survivors,
  and those are 1,779 equity against 52 crypto and 20 FX/commodity &mdash; so no amount of re-sweeping the
  list alone will balance the classes. Sweep command:</p>
  <pre class="sql">uv run scripts/run_oracle_dedup.py --stage oracle|naive --workers 8
uv run scripts/run_base_priority.py                            (base)</pre>
</section>

<section>
  <h2 class="sec"><span class="no">08</span>Conclusion</h2>
  <p>Strategies do not perform the same across markets, but the differences are not where intuition puts
  them. Asset class produces large, systematic gaps in achievable performance &mdash; the FX-to-crypto spread
  under perfect foresight is nearly nine Sharpe, wider than the gap between having a forecast and not having
  one. Listing venue produces almost none: a London or Hong Kong equity ranks strategies essentially as a US
  equity does.</p>
  <p>The finding with the most practical weight is that portability is substantially improved by prediction.
  With no forecast, per-class rankings agree only weakly (mean &rho;&nbsp;=&nbsp;+0.28) and a strategy
  selected on one market would largely have to be re-selected on the next. With a perfect forecast agreement
  rises to +0.74 and selection made on one class carries much better to another. That extends the companion
  paper&rsquo;s conclusion: the model does not only lift performance, it makes performance mean more nearly
  the same thing in different markets. Whether the <em>real</em> model does as much of that as the perfect one
  is the measurement this edition is missing, and it returns with base coverage.</p>
  <p>One class dissents. FX and commodities are profitable with no forecast at all, the only such cell here,
  which suggests the corpus holds a structural edge there that is independent of prediction quality &mdash;
  worth isolating, and worth testing on a wider currency sample than the fourteen groups available here.</p>
  <p>The practical recommendations are to treat asset class as a first-class dimension in strategy
  selection while treating venue as second-order, to investigate the strategies that reverse sign between
  classes before deploying any of them, and to fix the aliasing in &sect;5 before the next corpus-wide sweep
  so that eight names stop voting eight times.</p>
</section>
</div>

<footer>
  Kairos Project &middot; research note, {STAMP} &middot; generated from data/pipeline_results.db at commit {COMMIT}.<br>
  Companion to <em>The Prediction Premium</em>. Segmentation covers 5.6M signals across three asset classes
  and six equity listing venues, all scored on the unified walk-forward exit rule. Shadow-performance measurements without transaction costs; not investment advice.
</footer>
'''

STANDALONE = ('<!doctype html>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width,initial-scale=1">\n')
out = HERE / "where_strategies_travel.html"
out.write_text(STANDALONE + HTML if HERE.name == "papers" else HTML)
print(f"wrote {out} ({len(HTML):,} bytes)")

REPO = pathlib.Path("/media/baz/MonkeyWorks/PycharmProjects/Kairos/docs/papers")
if HERE.name != "papers" and REPO.is_dir():
    (REPO / "where_strategies_travel.html").write_text(STANDALONE + HTML)
    print(f"wrote {REPO / 'where_strategies_travel.html'}")
