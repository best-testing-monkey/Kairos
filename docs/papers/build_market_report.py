#!/usr/bin/env python3
"""Generate the 'Where Strategies Travel' market-segmentation report."""
import pathlib, json

HERE = pathlib.Path(__file__).parent

CLASSES = ["equity", "crypto", "fx_commodity"]
CNAME = {"equity": "Equities", "crypto": "Crypto", "fx_commodity": "FX &amp; commodities"}

# median Sharpe, positives, of-N, mean win %, signals, groups-per-strategy
LEVEL = {
    "oracle": {"equity": (2.401, 24, 38, 57.36, 2883844, 800),
               "crypto": (-4.784, 11, 38, 43.51, 314068, 68),
               "fx_commodity": (-1.905, 17, 38, 49.86, 202594, 28)},
    "base":   {"equity": (-0.006, 25, 50, 47.51, 997130, 206),
               "crypto": (0.137, 29, 50, 48.90, 1247385, 57),
               "fx_commodity": (0.296, 31, 50, 48.91, 432914, 20)},
    "naive":  {"equity": (-0.470, 12, 42, 47.95, 2597358, 736),
               "crypto": (-2.567, 3, 42, 32.11, 54061, 24)},
}
RANK = {
    "naive":  [("equity", "crypto", 0.161)],
    "base":   [("equity", "crypto", 0.700), ("equity", "fx_commodity", 0.712), ("crypto", "fx_commodity", 0.756)],
    "oracle": [("equity", "crypto", 0.702), ("equity", "fx_commodity", 0.800), ("crypto", "fx_commodity", 0.900)],
}
DIVERGENCE = {
    "oracle": [("volume_fade", -24.72, -50.81, -27.75), ("path_v_shape", 15.99, -3.54, 2.52),
               ("volume_confirmation", -6.21, -25.13, -18.24), ("cross_asset_spread", -5.18, -22.82, -14.12),
               ("rqa_determinism", 6.76, -8.65, -10.77), ("hurst_regime_switch", 11.99, -4.35, 3.19)],
    "base":   [("twap_execution", -3.89, 1.04, 9.35), ("pca_residual_reversal", 0.32, -1.62, 5.57),
               ("open_gap", 3.51, 8.71, 6.09), ("trend_following", -6.47, -3.94, -0.71),
               ("implementation_shortfall", -6.47, -3.75, -1.16), ("atr_bracket", -6.47, -3.45, -1.18)],
}
VENUE = {
    "oracle": [("United States", 44, 1.634, 2572445, None), ("Hong Kong", 38, 1.689, 71114, 0.961),
               ("Australia", 34, 4.636, 13433, 0.913), ("Germany", 34, 2.694, 11167, 0.965),
               ("Switzerland", 31, 3.423, 7935, 0.959), ("London", 31, 2.591, 6961, 0.966)],
    "naive":  [("United States", 44, -0.472, 2287443, None), ("Hong Kong", 38, -0.884, 70706, 0.607),
               ("Australia", 34, -0.096, 13323, 0.339), ("Germany", 34, -1.146, 11078, 0.641),
               ("Switzerland", 31, -0.369, 7887, 0.634), ("London", 31, -0.663, 6936, -0.039)],
    "base":   [("United States", 51, -0.001, 901557, None), ("Hong Kong", 29, -1.254, 13289, 0.624),
               ("London", 14, 0.975, 2903, 0.112), ("Germany", 15, -0.206, 2828, 0.218)],
}
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
        pairs = RANK[reg]
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
  computed on one common strategy set per regime so the coefficients are comparable. Without a forecast,
  a strategy&rsquo;s standing on equities barely predicts its standing on crypto. With one &mdash; real or
  perfect &mdash; the ordering largely transfers. Naive has only one comparable pair; FX/commodity coverage
  in that regime is too thin to score.</figcaption>
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
  but not across regimes, which cover different groups (see &sect;5).</caption>
  <thead><tr><th scope="col">Regime</th><th scope="col">Asset class</th>
    <th scope="col" class="num">Median Sharpe</th><th scope="col" class="num">Positive</th>
    <th scope="col" class="num">Mean win</th><th scope="col" class="num">Signals</th>
    <th scope="col" class="num">Groups/strat</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table></div>'''


def divergence_table(reg):
    rows = "".join(
        f'<tr><th scope="row" class="sname">{n}</th>'
        + "".join(f'<td class="num sh {"pos" if v>=0 else "neg"}"><span>{v:+.2f}</span></td>'
                  for v in (e, c, f))
        + f'<td class="num sub">{max(e,c,f)-min(e,c,f):.1f}</td></tr>'
        for n, e, c, f in DIVERGENCE[reg])
    return f'''<div class="tbl-wrap"><table class="tbl">
  <caption><b>Table {2 if reg=="oracle" else 3}.</b> Most class-dependent strategies under
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
  <caption><b>Table 5.</b> Effect of collapsing the eight aliased names to one on the companion
  paper&rsquo;s headline figures, matched sample.</caption>
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
  <i>&mdash;</i><span>29 August 2026</span></div>
  <h1>Where Strategies Travel</h1>
  <p class="standfirst">Does a strategy that works on equities work on crypto? On gold? On a Hong Kong
  listing? Segmenting 4.4&nbsp;million signals by asset class and listing venue &mdash; and finding that
  what makes performance portable is the forecast, not the market.</p>
  <div class="byline"><span>Kairos Project</span><span>1d bars &middot; 6-month window</span>
  <span>3 asset classes &middot; 6 listing venues</span><span>companion to <em>The Prediction Premium</em></span></div>
</header>

<div class="abstract">
  <h2>Abstract</h2>
  <p>The companion paper treated the 127-strategy corpus as one population. It is not. Segmenting results
  by asset class shows large, systematic differences in <em>level</em>: under perfect foresight the median
  strategy reaches +2.40 Sharpe on equities but &minus;4.78 on crypto, a gap far wider than the difference
  between prediction regimes.</p>
  <p>The more useful finding concerns <em>portability</em>. Ranking strategies within each asset class and
  correlating those rankings, the no-forecast regime shows almost no agreement between equities and crypto
  (&rho;&nbsp;=&nbsp;+0.16): which strategies appear to work on one class tells you almost nothing about the
  other. Add a forecast &mdash; the real model or a perfect one &mdash; and agreement rises to +0.70&ndash;0.90.
  Forecast information is what makes strategy quality a property of the strategy rather than of the market
  it happened to be tested on.</p>
  <p>Listing venue matters much less than asset class. Across six exchanges, strategy rankings agree with the
  US ordering at &rho;&nbsp;=&nbsp;+0.91 to +0.97 under oracle. A Hong Kong or German listing is, for these
  strategies, far more like a US listing than any equity is like a crypto pair.</p>
  <p>The segmentation also surfaced a defect: <strong>eight strategy names in the corpus are the same
  strategy</strong>, and correcting for it moves the companion paper&rsquo;s oracle median from +0.32 to +2.30.</p>
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
  <p>Without a forecast, they do not. Equity and crypto orderings agree at
  <strong>&rho;&nbsp;=&nbsp;+0.16</strong> &mdash; near-independent. A strategy that looks like the best of
  the corpus on equities is, under naive conditions, roughly a coin flip to be above median on crypto. This
  is consistent with the companion paper&rsquo;s finding that naive-regime performance is dominated by
  structural artefacts rather than by any transferable edge: artefacts of one market&rsquo;s bar geometry
  have no reason to reproduce in another.</p>
  <p>With a forecast, they largely do. Base reaches <strong>+0.70 to +0.76</strong> across all three class
  pairs and oracle <strong>+0.70 to +0.90</strong>. The forecast supplies a common signal that the strategies
  transform in their own characteristic ways, so a strategy&rsquo;s merit becomes a property of the
  transformation rather than of the market. That is the finding worth carrying forward:
  <strong>prediction is what makes strategy selection portable.</strong></p>
</section>

<section>
  <h2 class="sec"><span class="no">03</span>Level: markets are not equally tractable</h2>
  <p>Portability is not the same as parity. Rankings may agree while absolute performance differs sharply,
  and it does.</p>
</section>
</div>
<div class="wide">{level_table()}</div>
<div class="wrap">
<section>
  <p>Under perfect foresight the spread is dramatic: a median of <strong>+2.40 on equities against
  &minus;4.78 on crypto</strong>, with FX and commodities between them at &minus;1.91. The corpus extracts far
  more from a perfectly forecast equity bar than from a perfectly forecast crypto bar. The likely mechanism
  is bracket geometry rather than direction: crypto&rsquo;s intrabar range is wide relative to a
  strategy&rsquo;s stop distance, so a position that is directionally right is still stopped out on the way,
  and the single-bar evaluator books that as a loss. Crypto&rsquo;s mean win rate under oracle is 43.5%
  against equities&rsquo; 57.4% despite both having a perfect next-bar forecast &mdash; direction is not what
  is failing.</p>
  <p>Under the base model the classes converge, all three sitting near zero
  (&minus;0.01 equities, +0.14 crypto, +0.30 FX/commodities). Do not read this as crypto overtaking equities:
  the three regimes cover different group samples, so a level in one regime is not comparable to a level in
  another. Checked properly &mdash; on the 1,424 matched (group, strategy) cells where oracle and base both
  ran on crypto &mdash; oracle averages <strong>+4.97</strong> against base&rsquo;s <strong>+0.63</strong>. The
  ceiling still towers over the system everywhere; only the unmatched view suggested otherwise.</p>
  <h3>3.1 &mdash; Which strategies are most market-dependent</h3>
</section>
</div>
<div class="wide">{divergence_table("oracle")}{divergence_table("base")}</div>
<div class="wrap">
<section>
  <p>Two patterns stand out. <code>volume_fade</code> and <code>volume_confirmation</code> degrade most on
  crypto &mdash; both read volume, and crypto volume is reported inconsistently across venues and is not
  comparable to a consolidated equity tape. <code>path_v_shape</code>, <code>rqa_determinism</code> and
  <code>hurst_regime_switch</code> are strongly equity-specific under oracle, all three depending on
  mean-reverting intraday path shape that crypto&rsquo;s trendier, thinner sessions do not supply.</p>
  <p>Under the base model, <code>twap_execution</code> genuinely reverses sign &mdash; &minus;3.89 on equities,
  +9.35 on FX and commodities &mdash; which is what a market-specific deployment rule would have to catch.
  <code>open_gap</code> is the encouraging counter-case: positive on all three classes and strongest on
  crypto (+8.71), a strategy whose premise travels.</p>
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
  (<strong>&rho;&nbsp;=&nbsp;+0.91 to +0.97</strong>). Whatever differences exist between trading in New York,
  Hong Kong, Frankfurt, Sydney, Zurich and London, they do not change which strategies work. Compare that to
  the +0.70 equity-versus-crypto figure and the ordering is clear: <strong>instrument type is a first-order
  effect; listing venue is a second-order one.</strong></p>
  <p>Non-US venues do show higher median Sharpe under oracle (+1.7 to +4.6 against the US +1.6), but this
  should be discounted heavily. Non-US signal counts are two orders of magnitude smaller (7,000&ndash;71,000
  against 2.57 million), and the universe screen&rsquo;s flat $50M liquidity floor is far more selective
  abroad than in the US &mdash; the surviving non-US names are their markets&rsquo; largest and most liquid,
  which is a different population, not a better market. Under naive the same venues show scattered, weak
  agreement (&rho; from &minus;0.04 to +0.64), reinforcing &sect;2: strip the forecast and cross-market
  agreement collapses at venue level too.</p>
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
  <p>Counts of profitable strategies are unaffected (all eight are negative everywhere, so none was ever
  counted as a success), but the denominators and medians in the companion paper should be read as
  <strong>44 distinct strategies, not 51</strong>. The oracle median in particular is understated by a factor
  of seven. The corpus-level conclusions do not reverse &mdash; naive stays overwhelmingly unprofitable, base
  stays broadly positive, oracle stays far above both &mdash; but the published oracle median is wrong and
  should be corrected.</p>
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
    <li><strong>The exit-rule asymmetry carries over.</strong> As in the companion paper, oracle and base are
    scored one bar ahead then force-closed while naive walks forward multi-bar. This bears directly on
    &sect;3&rsquo;s crypto explanation: a single-bar forced close penalises wide-range instruments more than a
    multi-bar walk-forward would, so part of crypto&rsquo;s oracle deficit may be evaluator artefact rather
    than market difficulty.</li>
    <li><strong>Class samples are very unequal.</strong> Equities contribute 800 groups per strategy under
    oracle against 68 for crypto and 28 for FX/commodities. The crypto and FX medians rest on far less data,
    and no confidence intervals are computed anywhere in this note.</li>
    <li><strong>Venue coverage under base is thin.</strong> London and Germany carry 14 and 15 strategies on
    under 3,000 signals each; their &rho;-versus-US figures (+0.11, +0.22) are too small a sample to
    distinguish from the oracle regime&rsquo;s much higher agreement. The venue conclusion rests mainly on the
    oracle row.</li>
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

SELECT strategy_name, sharpe, signal_count, win_rate, avg_pnl_per_trade, assets
FROM   oracle_results          -- or model_results for base
WHERE  stage = ?  AND signal_count &gt;= 3;</pre>
  <p>Per-strategy figures are signal-count-weighted within each class, requiring at least 5 groups per
  strategy per class. Rank correlations use one common strategy set per regime, so the coefficients within a
  regime are comparable to each other. Scripts: <code>analyze_by_market3.py</code> (analysis),
  <code>build_market_report.py</code> (this page).</p>
</section>

<section>
  <h2 class="sec"><span class="no">08</span>Conclusion</h2>
  <p>Strategies do not perform the same across markets, but the differences are not where intuition puts
  them. Asset class produces large, systematic gaps in achievable performance &mdash; the equity-to-crypto
  spread under perfect foresight is wider than the gap between having a forecast and not having one. Listing
  venue produces almost none: a London or Hong Kong equity ranks strategies essentially as a US equity does.</p>
  <p>The finding with the most practical weight is that portability is created by prediction. With no
  forecast, per-class rankings are nearly unrelated and any strategy selected on one market would have to be
  re-selected on the next. With a forecast, rankings converge, and selection made on one class carries to
  another. That extends the companion paper&rsquo;s conclusion: the model does not only lift performance, it
  makes performance mean the same thing in different markets.</p>
  <p>The practical recommendations are to treat asset class as a first-class dimension in strategy
  selection while treating venue as second-order, to investigate the strategies that reverse sign between
  classes before deploying any of them, and to fix the aliasing in &sect;5 before the next corpus-wide sweep
  so that eight names stop voting eight times.</p>
</section>
</div>

<footer>
  Kairos Project &middot; research note, 29 August 2026 &middot; generated from data/pipeline_results.db.<br>
  Companion to <em>The Prediction Premium</em>. Segmentation covers 4.4M+ signals across three asset classes
  and six equity listing venues. Shadow-performance measurements without transaction costs; not investment advice.
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
