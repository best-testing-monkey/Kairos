#!/usr/bin/env python3
"""Generate the Prediction Premium whitepaper HTML from the matched results table."""
import json, pathlib

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

data = json.loads((HERE / "paper_table.json").read_text())
rows = data["rows"]

STAGES = ("naive", "base", "oracle")

# --- box-plot geometry -------------------------------------------------------
# Derived from the same rows the full table renders, never transcribed: hand-kept
# constants previously carried 2dp values into 3dp table cells, printing a median
# of +0.160 where the data said +0.159.
def _median(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

def _box(stage):
    sh = sorted(r["sh"][stage] for r in rows)
    tot = sum(r["n"][stage] for r in rows)
    return dict(
        mn=sh[0], q1=sh[len(sh) // 4], med=_median(sh),
        q3=sh[3 * len(sh) // 4], mx=sh[-1],
        n=tot,
        pos=sum(1 for v in sh if v > 0),
        win=sum(r["win"][stage] * r["n"][stage] for r in rows) / tot * 100,
        pnl=_median([r["pnl"][stage] for r in rows]) * 100,
    )

BOX = {s: _box(s) for s in STAGES}
NSTRAT = len(rows)
NGROUPS = data["matched_groups"]
AX_LO, AX_HI = -6.0, 12.0
def pct(v):
    return max(0.0, min(100.0, (v - AX_LO) / (AX_HI - AX_LO) * 100.0))

LABEL = {"naive": "Naive", "base": "Base model", "oracle": "Oracle"}
SUB = {"naive": "no prediction", "base": "Kronos-base", "oracle": "perfect foresight"}

def figure():
    ticks = [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
    tick_html = "".join(
        f'<span class="tick{" tick-zero" if t == 0 else ""}" style="left:{pct(t):.3f}%">{t}</span>'
        for t in ticks)
    grid_html = "".join(
        f'<i class="gl{" gl-zero" if t == 0 else ""}" style="left:{pct(t):.3f}%"></i>'
        for t in ticks)
    rowhtml = []
    for s in STAGES:
        b = BOX[s]
        l, r = pct(b["q1"]), pct(b["q3"])
        rowhtml.append(f'''
      <div class="bp-row" data-stage="{s}">
        <div class="bp-label"><b>{LABEL[s]}</b><span>{SUB[s]}</span></div>
        <div class="bp-track">
          {grid_html}
          <i class="whisk" style="left:0;width:{l:.3f}%"></i>
          <i class="whisk" style="left:{r:.3f}%;right:0"></i>
          <i class="box" style="left:{l:.3f}%;width:{max(r - l, 0.6):.3f}%"></i>
          <i class="med" style="left:{pct(b["med"]):.3f}%"></i>
          <span class="cap cap-lo">{b["mn"]:+.1f}</span>
          <span class="cap cap-hi">{b["mx"]:+.1f}</span>
        </div>
        <div class="bp-med">{b["med"]:+.2f}</div>
      </div>''')
    return f'''<figure class="fig">
  <div class="bp">
    {"".join(rowhtml)}
    <div class="bp-axis"><div class="bp-label"></div><div class="bp-track">{tick_html}</div><div class="bp-med"></div></div>
  </div>
  <figcaption><b>Figure 1.</b> Distribution of per-strategy signal-weighted Sharpe across the {NSTRAT} distinct strategies
  evaluated in all three regimes. Boxes span the first to third quartile; the vertical rule is the median;
  whiskers are clipped at the axis bounds with the true extremes printed at each end. The naive box sits
  entirely left of zero. The base model's median crosses it. Oracle's box runs off the scale at
  {BOX["oracle"]["q3"]:+.2f}, on a true maximum of {BOX["oracle"]["mx"]:+.2f}.</figcaption>
</figure>'''

def summary_table():
    cells = []
    for s in STAGES:
        b = BOX[s]
        cells.append(f'''<tr data-stage="{s}">
      <th scope="row"><span class="swatch"></span>{LABEL[s]}</th>
      <td class="num">{b["n"]:,}</td>
      <td class="num">{b["pos"]} / {NSTRAT}</td>
      <td class="num">{b["med"]:+.3f}</td>
      <td class="num">{b["q1"]:+.2f}</td>
      <td class="num">{b["q3"]:+.2f}</td>
      <td class="num">{b["win"]:.2f}%</td>
      <td class="num">{b["pnl"]:+.4f}%</td>
    </tr>''')
    return f'''<div class="tbl-wrap">
<table class="tbl">
  <caption><b>Table 1.</b> Aggregate results by regime, matched sample.</caption>
  <thead><tr>
    <th scope="col">Regime</th><th scope="col" class="num">Signals</th>
    <th scope="col" class="num">Sharpe &gt; 0</th><th scope="col" class="num">Median Sharpe</th>
    <th scope="col" class="num">Q1</th><th scope="col" class="num">Q3</th>
    <th scope="col" class="num">Mean win rate</th><th scope="col" class="num">Median return/trade</th>
  </tr></thead>
  <tbody>{"".join(cells)}</tbody>
</table></div>'''

def full_table():
    span = 12.0  # bar half-width scale, Sharpe units
    body = []
    for r in rows:
        tds = []
        for s in STAGES:
            v = r["sh"][s]
            w = min(abs(v) / span, 1.0) * 50.0
            side = "pos" if v >= 0 else "neg"
            style = (f"left:50%;width:{w:.2f}%" if v >= 0
                     else f"right:50%;width:{w:.2f}%")
            tds.append(
                f'<td class="num sh {side}" data-stage="{s}">'
                f'<i class="bar" style="{style}"></i><span>{v:+.2f}</span></td>')
        wins = "".join(f'<td class="num sub">{r["win"][s] * 100:.1f}</td>' for s in STAGES)
        body.append(
            f'<tr><th scope="row" class="sname">{r["name"]}</th>'
            f'{"".join(tds)}{wins}'
            f'<td class="num sub">{r["n"]["base"]:,}</td></tr>')
    return f'''<div class="tbl-wrap tbl-tall">
<table class="tbl tbl-full">
  <caption><b>Table 2.</b> Per-strategy signal-weighted Sharpe and win rate in all three regimes,
  ordered by base-model Sharpe. Bars are scaled to &plusmn;12 Sharpe and clipped beyond it.</caption>
  <thead>
    <tr>
      <th scope="col" rowspan="2">Strategy</th>
      <th scope="col" colspan="3" class="grp">Signal-weighted Sharpe</th>
      <th scope="col" colspan="3" class="grp">Win rate %</th>
      <th scope="col" rowspan="2" class="num">Base signals</th>
    </tr>
    <tr>
      <th scope="col" class="num sh-h" data-stage="naive">Naive</th>
      <th scope="col" class="num sh-h" data-stage="base">Base</th>
      <th scope="col" class="num sh-h" data-stage="oracle">Oracle</th>
      <th scope="col" class="num">Naive</th><th scope="col" class="num">Base</th><th scope="col" class="num">Oracle</th>
    </tr>
  </thead>
  <tbody>{"".join(body)}</tbody>
</table></div>'''


HTML = f'''<title>The Prediction Premium</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@450;600;700&family=Spectral:ital,wght@0,400;0,500;1,400&display=swap">
<style>
:root {{
  --ground:#FAFBFC; --surface:#EFF3F6; --surface-2:#E4EAEF;
  --ink:#0E141A; --ink-mid:#3C4854; --ink-mute:#66727E; --rule:#D3DBE2;
  --naive:#3F6F6A; --base:#1E51A0; --oracle:#8A6009;
  --naive-soft:#3F6F6A22; --base-soft:#1E51A022; --oracle-soft:#8A600922;
  --accent:#1E51A0;
  --measure:68ch;
  --f-head:"IBM Plex Sans Condensed",-apple-system,"Segoe UI",sans-serif;
  --f-body:"Spectral",Georgia,"Times New Roman",serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0E1317; --surface:#161D23; --surface-2:#1E272F;
    --ink:#E3E9EF; --ink-mid:#B3BFCA; --ink-mute:#7D8B98; --rule:#2A343D;
    --naive:#6FB8AC; --base:#7EA9EC; --oracle:#DDB055;
    --naive-soft:#6FB8AC26; --base-soft:#7EA9EC26; --oracle-soft:#DDB05526;
    --accent:#7EA9EC;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0E1317; --surface:#161D23; --surface-2:#1E272F;
  --ink:#E3E9EF; --ink-mid:#B3BFCA; --ink-mute:#7D8B98; --rule:#2A343D;
  --naive:#6FB8AC; --base:#7EA9EC; --oracle:#DDB055;
  --naive-soft:#6FB8AC26; --base-soft:#7EA9EC26; --oracle-soft:#DDB05526;
  --accent:#7EA9EC;
}}

* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--ink);
  font-family:var(--f-body); font-size:17px; line-height:1.62;
  margin:0; padding:0 5vw 8rem;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:var(--measure); margin:0 auto; }}
.wide {{ max-width:min(1100px,94vw); margin-inline:auto; }}

/* ---- masthead ---- */
header.mast {{ max-width:min(1100px,94vw); margin:0 auto; padding:5.5rem 0 0; }}
.eyebrow {{
  font-family:var(--f-head); font-weight:600; font-size:.74rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ink-mute);
  display:flex; gap:.7rem; align-items:center; flex-wrap:wrap;
}}
.eyebrow i {{ font-style:normal; color:var(--rule); }}
h1 {{
  font-family:var(--f-head); font-weight:700; font-size:clamp(2.6rem,7vw,4.6rem);
  line-height:1.02; letter-spacing:-.018em; margin:1.1rem 0 0; text-wrap:balance;
}}
.standfirst {{
  font-size:clamp(1.08rem,2.2vw,1.32rem); line-height:1.5; color:var(--ink-mid);
  max-width:56ch; margin:1.4rem 0 0; font-style:italic;
}}
.byline {{
  font-family:var(--f-mono); font-size:.78rem; color:var(--ink-mute);
  margin:2.2rem 0 0; padding-top:1.1rem; border-top:1px solid var(--rule);
  display:flex; gap:1.6rem; flex-wrap:wrap;
}}

/* ---- abstract ---- */
.abstract {{
  background:var(--surface); border-left:3px solid var(--accent);
  padding:1.6rem 1.9rem; margin:3.2rem auto 0; max-width:min(1100px,94vw);
}}
.abstract h2 {{
  font-family:var(--f-head); font-size:.75rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-mute); margin:0 0 .7rem; font-weight:600;
}}
.abstract p {{ margin:0 0 .85rem; font-size:1rem; }}
.abstract p:last-child {{ margin-bottom:0; }}

/* ---- sections ---- */
section {{ margin:4.2rem auto 0; }}
h2.sec {{
  font-family:var(--f-head); font-weight:600; font-size:1.72rem; letter-spacing:-.008em;
  margin:0 0 1.1rem; display:flex; gap:.85rem; align-items:baseline; text-wrap:balance;
}}
h2.sec .no {{
  font-family:var(--f-mono); font-size:.86rem; color:var(--accent);
  font-weight:500; flex:none; padding-top:.15em;
}}
h3 {{
  font-family:var(--f-head); font-weight:600; font-size:1.1rem;
  margin:2.3rem 0 .6rem; color:var(--ink);
}}
p {{ margin:0 0 1.05rem; }}
a {{ color:var(--accent); }}
strong {{ font-weight:500; color:var(--ink); }}
code, .mono {{ font-family:var(--f-mono); font-size:.86em; }}
code {{ background:var(--surface-2); padding:.1em .35em; border-radius:2px; }}
ul, ol {{ margin:0 0 1.05rem; padding-left:1.4rem; }}
li {{ margin-bottom:.5rem; }}
.lede::first-letter {{
  font-family:var(--f-head); font-weight:700; font-size:3.1em; float:left;
  line-height:.82; padding:.06em .09em 0 0; color:var(--accent);
}}

/* ---- regime cards ---- */
.regimes {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); margin:1.8rem 0 2rem; }}
.rg {{ background:var(--surface); padding:1.15rem 1.25rem; border-top:3px solid var(--c); }}
.rg[data-stage="naive"] {{ --c:var(--naive); }}
.rg[data-stage="base"] {{ --c:var(--base); }}
.rg[data-stage="oracle"] {{ --c:var(--oracle); }}
.rg h4 {{ font-family:var(--f-head); font-size:1.02rem; font-weight:600; margin:0 0 .1rem; color:var(--c); }}
.rg .flag {{ font-family:var(--f-mono); font-size:.74rem; color:var(--ink-mute); display:block; margin-bottom:.6rem; }}
.rg p {{ margin:0; font-size:.93rem; line-height:1.5; color:var(--ink-mid); }}

/* ---- figure / boxplot ---- */
.fig {{ margin:2.4rem 0 2rem; }}
.bp {{ background:var(--surface); padding:1.6rem 1.4rem 1rem; }}
.bp-row, .bp-axis {{ display:grid; grid-template-columns:minmax(96px,132px) 1fr 62px; align-items:center; gap:.9rem; }}
.bp-row {{ padding:.55rem 0; }}
.bp-row[data-stage="naive"] {{ --c:var(--naive); --cs:var(--naive-soft); }}
.bp-row[data-stage="base"] {{ --c:var(--base); --cs:var(--base-soft); }}
.bp-row[data-stage="oracle"] {{ --c:var(--oracle); --cs:var(--oracle-soft); }}
.bp-label b {{ font-family:var(--f-head); font-weight:600; font-size:.95rem; display:block; color:var(--c); }}
.bp-label span {{ font-family:var(--f-mono); font-size:.68rem; color:var(--ink-mute); }}
.bp-track {{ position:relative; height:38px; }}
.bp-track i {{ position:absolute; display:block; }}
.gl {{ top:0; bottom:0; width:1px; background:var(--rule); }}
.gl-zero {{ background:var(--ink-mute); width:1px; opacity:.65; }}
.whisk {{ top:50%; height:1px; background:var(--c); opacity:.5; transform:translateY(-50%); }}
.box {{ top:7px; bottom:7px; height:auto; background:var(--cs); border:1.5px solid var(--c); }}
.med {{ top:3px; bottom:3px; height:auto; width:2.5px; background:var(--c); }}
.cap {{ position:absolute; top:50%; transform:translateY(-50%); font-family:var(--f-mono); font-size:.66rem; color:var(--ink-mute); }}
.cap-lo {{ left:2px; }} .cap-hi {{ right:2px; }}
.bp-med {{ font-family:var(--f-mono); font-size:.86rem; font-weight:500; color:var(--c); text-align:right; font-variant-numeric:tabular-nums; }}
.bp-axis {{ border-top:1px solid var(--rule); margin-top:.5rem; padding-top:.35rem; }}
.bp-axis .bp-track {{ height:18px; }}
.tick {{ position:absolute; top:0; transform:translateX(-50%); font-family:var(--f-mono); font-size:.68rem; color:var(--ink-mute); }}
.tick-zero {{ color:var(--ink); font-weight:500; }}
figcaption {{ font-size:.88rem; line-height:1.5; color:var(--ink-mute); margin-top:.9rem; max-width:78ch; }}
figcaption b {{ color:var(--ink-mid); font-weight:500; }}

/* ---- tables ---- */
.tbl-wrap {{ overflow-x:auto; margin:1.8rem 0; background:var(--surface); }}
.tbl-tall {{ max-height:76vh; overflow-y:auto; }}
table.tbl {{ border-collapse:collapse; width:100%; font-family:var(--f-mono); font-size:.8rem; }}
.tbl caption {{
  font-family:var(--f-body); font-size:.88rem; color:var(--ink-mute); text-align:left;
  padding:1rem 1.1rem; line-height:1.45; caption-side:top;
}}
.tbl caption b {{ color:var(--ink-mid); font-weight:500; }}
.tbl th, .tbl td {{ padding:.42rem .7rem; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap; }}
.tbl thead th {{
  position:sticky; top:0; z-index:2; background:var(--surface-2);
  font-family:var(--f-head); font-weight:600; font-size:.76rem;
  letter-spacing:.04em; color:var(--ink-mid);
}}
.tbl-full thead tr:nth-child(2) th {{ top:29px; }}
.tbl .grp {{ text-align:center; border-bottom:1px solid var(--rule); }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.sub {{ color:var(--ink-mute); }}
.sname {{ font-weight:400; color:var(--ink); }}
.swatch {{ display:inline-block; width:9px; height:9px; margin-right:.55rem; background:var(--c); }}
[data-stage="naive"] {{ --c:var(--naive); }}
[data-stage="base"] {{ --c:var(--base); }}
[data-stage="oracle"] {{ --c:var(--oracle); }}
.sh {{ position:relative; min-width:96px; }}
.sh .bar {{ position:absolute; top:4px; bottom:4px; background:var(--c); opacity:.24; }}
.sh span {{ position:relative; }}
.sh.neg span {{ color:var(--ink-mute); }}
.sh-h {{ color:var(--c); }}
tbody tr:hover td, tbody tr:hover th {{ background:var(--surface-2); }}

/* ---- callout ---- */
.callout {{
  border:1px solid var(--rule); border-left:3px solid var(--oracle);
  background:var(--surface); padding:1.3rem 1.5rem; margin:2rem 0;
}}
.callout h4 {{ font-family:var(--f-head); font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; color:var(--oracle); margin:0 0 .6rem; font-weight:600; }}
.callout p:last-child {{ margin-bottom:0; }}

/* ---- provenance ---- */
.prov {{ font-family:var(--f-mono); font-size:.8rem; background:var(--surface); padding:1.2rem 1.4rem; margin:1.4rem 0; overflow-x:auto; }}
.prov dt {{ color:var(--accent); margin-top:.9rem; }}
.prov dt:first-child {{ margin-top:0; }}
.prov dd {{ margin:.15rem 0 0; color:var(--ink-mid); white-space:pre; }}
pre.sql {{
  font-family:var(--f-mono); font-size:.78rem; line-height:1.55; background:var(--surface);
  padding:1.1rem 1.3rem; overflow-x:auto; margin:1.2rem 0; color:var(--ink-mid);
  border-left:3px solid var(--rule);
}}

footer {{
  max-width:min(1100px,94vw); margin:5rem auto 0; padding-top:1.4rem;
  border-top:1px solid var(--rule); font-family:var(--f-mono);
  font-size:.75rem; color:var(--ink-mute); line-height:1.7;
}}
@media (max-width:640px) {{
  body {{ font-size:16px; }}
  .bp-row, .bp-axis {{ grid-template-columns:74px 1fr 46px; gap:.5rem; }}
  .cap {{ display:none; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<header class="mast">
  <div class="eyebrow">
    <span>Kairos Research Note</span><i>&mdash;</i><span>Strategy Evaluation</span><i>&mdash;</i><span>31 August 2026</span>
  </div>
  <h1>The Prediction Premium</h1>
  <p class="standfirst">How much of a trading strategy&rsquo;s measured edge comes from the strategy,
  and how much comes from the forecast it is fed? A three-regime experiment, re-run end to end on a
  single exit rule.</p>
  <div class="byline">
    <span>Kairos Project</span>
    <span>1d bars &middot; 6-month window</span>
    <span>127 strategies &middot; {NGROUPS} matched asset groups</span>
    <span>commit e8c596a</span>
  </div>
</header>

<div class="abstract">
  <h2>Abstract</h2>
  <p>Backtest results conflate two separable things: the merit of a strategy&rsquo;s decision rule, and the
  quality of the forecast that rule consumes. We separate them by running the same 127-strategy corpus
  over the same assets and the same period under three prediction regimes &mdash; a <strong>naive</strong> floor
  with no forecast information, an <strong>oracle</strong> ceiling with perfect foresight of the next bar, and the
  <strong>base</strong> regime driven by the pretrained Kronos forecasting model.</p>
  <p>All three regimes are now scored by one walk-forward exit rule, correcting the asymmetry the first
  edition of this note listed as its largest caveat (&sect;8.1). On a matched sample of {NGROUPS} asset groups and
  the {NSTRAT} distinct strategies that produced signals in all three regimes, the results separate cleanly.
  Stripped of forecast information, only {BOX["naive"]["pos"]} of {NSTRAT} strategies hold a positive
  signal-weighted Sharpe and the median strategy sits at {BOX["naive"]["med"]:+.2f}. Given perfect foresight,
  the median rises to {BOX["oracle"]["med"]:+.2f} and the upper quartile to {BOX["oracle"]["q3"]:+.2f} &mdash;
  but {NSTRAT - BOX["oracle"]["pos"]} of {NSTRAT} still lose money knowing the future. The base model sits
  between them, lifting {BOX["base"]["pos"]} of {NSTRAT} strategies above zero to a median of
  {BOX["base"]["med"]:+.2f} &mdash; most of the way to the oracle on <em>breadth</em>, and a small fraction of
  the way on <em>magnitude</em>.</p>
  <p>The three-regime sample is thin, and honestly so: the base sweep is being re-run on the new exit rule and
  has reached {NGROUPS} of 961 groups, so the matched sample is a fraction of its predecessor and is ordered by
  oracle Sharpe rather than drawn at random (&sect;3.3, &sect;6). The oracle and naive sweeps are complete and
  exactly paired at 961 groups; where a claim needs only those two, we report it on the full corpus.</p>
  <p>We conclude that forecast quality is the dominant term in this corpus&rsquo;s measured performance, that
  a substantial subset of strategies is unsalvageable by any forecast, and that the pretrained model
  supplies real, measurable directional information &mdash; without yet supplying conviction.</p>
</div>

<div class="wrap">

<section>
  <h2 class="sec"><span class="no">01</span>The question</h2>
  <p class="lede">A backtest reports one number where two questions are hiding. When a momentum rule
  posts a Sharpe of 1.4, that figure is the product of a decision rule and the information the rule was
  given. Change either and the number moves. Most evaluation frameworks hold the information fixed and
  vary the rule, which answers &ldquo;which strategy is best?&rdquo; but never &ldquo;is any of this
  coming from the strategy at all?&rdquo;</p>
  <p>That second question matters more than it sounds. If a strategy&rsquo;s edge survives the removal of
  all forecast information, the edge is structural &mdash; it lives in the execution rule, the bracket
  geometry, or an exploitable regularity in the bar data itself. If it collapses, the strategy is a
  transmission mechanism, and its performance is a measurement of the forecaster wearing the
  strategy&rsquo;s clothes.</p>
  <p>Kairos couples a corpus of 127 strategy implementations to Kronos, a transformer that produces a
  sampled distribution over the next bar. This note asks how the corpus performs when that coupling is
  cut, when it is replaced with perfect information, and when it is left intact.</p>
</section>

<section>
  <h2 class="sec"><span class="no">02</span>Three regimes</h2>
  <p>Each regime changes exactly one thing: what the strategy knows about the next bar when it decides.
  Everything else &mdash; the asset groups, the calendar, the filters, the strategy code &mdash; is held constant.</p>

  <div class="regimes">
    <div class="rg" data-stage="naive">
      <h4>Naive</h4><span class="flag">--naive-baseline &middot; floor</span>
      <p>The most recent bar is withheld from the history the strategy sees and handed back as the
      forecast, asserted with near-certainty: the next bar is predicted to repeat the last completed one.
      Nothing about the future is used anywhere. That bar has closed by the time the trade is placed, entry
      is its close, and only strictly later bars resolve it.</p>
    </div>
    <div class="rg" data-stage="base">
      <h4>Base model</h4><span class="flag">NeoQuasar/Kronos-base &middot; system under test</span>
      <p>The production path. A pretrained Kronos transformer emits 100 sampled next-bar trajectories;
      the strategy sees the resulting distribution and decides from it. No fine-tuning, no
      asset-specific adaptation.</p>
    </div>
    <div class="rg" data-stage="oracle">
      <h4>Oracle</h4><span class="flag">--no-prediction &middot; ceiling</span>
      <p>The model&rsquo;s distribution is replaced with the actual next bar&rsquo;s OHLCV. The strategy
      decides with perfect foresight one step ahead. Not achievable &mdash; it exists to bound what the
      decision rule could extract given a flawless forecast.</p>
    </div>
  </div>

  <p>The naive regime deserves a note, because two natural constructions of it are wrong in opposite
  directions. The first fed the distribution-building step the <em>current</em> bar as both the centre and
  the shape of the forecast. That is not a neutral no-information test: it silently asserts zero drift,
  which centres every distribution exactly on the entry price and handicaps every directional strategy by
  construction. A full sweep was run on that version before the flaw was caught, and its results were
  discarded. The second, which produced the previous edition of this note, took the oracle&rsquo;s decision
  unchanged and corrected only the accounting, re-anchoring entry to the bar the oracle had peeked at. The
  arithmetic there is sound &mdash; foreseeing a bar and then entering at its close is worth nothing, and
  the re-anchoring cancels the advantage exactly &mdash; but the decision still read a bar that had not
  happened, so the regime was free of lookahead by cancellation rather than by construction. Withholding
  fixes both: the forecast bar is one the strategy could genuinely have held, and because it is withheld,
  the distribution&rsquo;s centre stays distinct from its shape.</p>
</section>

<section>
  <h2 class="sec"><span class="no">03</span>Experimental design</h2>

  <h3>3.1 &mdash; Universe screening</h3>
  <p>The candidate universe is a scraped symbol list spanning US and international equities,
  crypto, FX and commodities. At the daily interval, <strong>38,445 symbols were screened</strong> on three
  criteria, all evaluated on the symbol&rsquo;s own trading calendar:</p>
  <ul>
    <li><strong>History:</strong> at least 300 bars available &mdash; the model&rsquo;s own lookback window,
    so a symbol that cannot fill one is not evaluable at all.</li>
    <li><strong>Liquidity:</strong> median daily dollar volume of at least $50M for equities and $10M for
    crypto, currency-converted to USD for non-USD listings. FX pairs are exempt &mdash; the data source
    reports no meaningful volume for them &mdash; and are flagged rather than dropped.</li>
    <li><strong>Volatility:</strong> ATR of at least 0.5% of the last close, which removes instruments too
    inert for a bracketed intraday-to-daily strategy to act on at all.</li>
  </ul>
  <p><strong>1,851 symbols passed.</strong> These survivors are the only assets that appear anywhere in
  this study.</p>

  <h3>3.2 &mdash; Why assets are grouped, and on what criterion</h3>
  <p>Grouping is not a diversification or risk-management choice. It is a functional prerequisite:
  a subset of the corpus &mdash; <code>cross_asset_rank</code>, <code>cross_asset_spread</code>,
  <code>cross_asset_momentum</code> and their relatives &mdash; are <strong>cross-asset strategies that
  cannot produce a signal from a single instrument</strong>. They rank, spread, or transfer momentum
  <em>between</em> instruments, and they only mean anything when handed a basket whose members actually
  move together. A basket of unrelated symbols makes a cross-asset rank arbitrary. So the pipeline builds
  baskets of co-moving symbols and evaluates the whole corpus on them; the single-asset strategies run on
  the same groups as a consequence of that decision, not as a reason for it.</p>
  <p>The co-movement criterion is <strong>correlation of daily log returns</strong>, not of price levels
  &mdash; correlating raw prices between two trending series produces high coefficients that carry no
  information about shared behaviour. For each pair, closes over a 400-bar window are aligned, converted to
  log returns, and scored by full-window Pearson correlation, with a 30-bar rolling median retained
  alongside it as a stability check. A pair needs at least 150 overlapping bars to be scored at all.
  <strong>1,697,403 pairs were computed</strong> across the survivor set.</p>
  <p>A pair qualifies for grouping on <strong>absolute</strong> correlation above a per-asset-class
  threshold &mdash; 0.60 by default and 0.75 for crypto, with the stricter of the two symbols&rsquo; class
  thresholds applying to a cross-class pair. Because the test is on |&rho;|, a consistently
  <em>inversely</em> moving pair qualifies as readily as a positively correlated one; both give the
  cross-asset strategies the stable relationship they need, and observed group-level mean correlations
  run from &minus;0.997 to +0.9998, averaging +0.54.</p>
  <p>Qualifying pairs are clustered greedily: pairs are sorted by |&rho;| descending and each is either
  absorbed into an existing group that already contains one of its symbols and has capacity, or seeds a new
  group, to a <strong>maximum of four symbols</strong>. Membership overlaps, so a popular symbol can appear
  in many groups. This produced <strong>13,472 candidate groups</strong> &mdash; 13,103 equity, 271
  cross-class, 94 crypto, 4 FX/commodity.</p>
  <p>That overlap is enormously redundant for backtesting: a single liquid symbol can occur in dozens of
  near-duplicate four-symbol combinations, and testing each one re-measures the same assets. The candidates
  are therefore reduced by <strong>greedy set cover</strong> &mdash; repeatedly take the group covering the
  most not-yet-covered symbols &mdash; to <strong>961 groups of one to four assets, leaving zero survivor
  symbols uncovered</strong>. All three sweeps in this study draw from that same 961-group list.</p>
  <p>All runs use daily bars over a six-month backtest window with 100 prediction samples per bar.</p>

  <h3>3.3 &mdash; Matched sample</h3>
  <p><strong>Only rows scored by the current exit rule count.</strong> Every result here comes from the
  re-sweep that followed the exit-rule unification of 29 August 2026 (&sect;3.4, &sect;8.1). Rows written before
  it were scored one bar ahead and force-closed, which is a different measurement, so they are excluded
  outright rather than blended in. Oracle and naive are complete: <strong>961 of 961 groups</strong> each, the
  same 961, so those two regimes are exactly paired for the first time. The base sweep is still running and
  has reached <strong>{NGROUPS} groups</strong>.</p>
  <p><strong>Matched sample.</strong> Comparing regimes on unequal footprints would confound regime with asset
  mix. Every three-regime figure in this note is therefore computed on the <strong>{NGROUPS} groups present in all three sweeps</strong>
  (matched on assets, interval and backtest period), restricted further to the <strong>{NSTRAT} distinct
  strategies that fired signals in all three regimes</strong> &mdash; {BOX["naive"]["n"] + BOX["base"]["n"] + BOX["oracle"]["n"]:,} signals in all. This is the paper&rsquo;s central
  methodological commitment: all comparisons are paired. It costs sample size here &mdash; the previous
  edition matched 451 groups &mdash; and the base sweep is ordered by oracle Sharpe rather than shuffled, so
  the {NGROUPS} groups over-represent assets the oracle did well on. Both effects are visible in the results and
  are quantified in &sect;4.4.</p>
  <p><strong>Aliased strategies are collapsed.</strong> Eight strategy names in the corpus are the same
  strategy: <code>trend_following</code> and seven filters that wrap it
  (<code>cds_spread_filter</code>, <code>cot_positioning_filter</code>, <code>dark_pool_filter</code>,
  <code>fractal_dimension</code>, <code>gaussian_process</code>, <code>insider_cluster</code>,
  <code>onchain_flow_filter</code>) and gate on context keys the pipeline never supplies, so every gate
  defaults false and each is an unmodified pass-through. Counted separately they would contribute seven
  redundant copies of one behaviour to every median. Only <code>trend_following</code> is retained.</p>
  <h3>3.4 &mdash; Aggregation and evaluation</h3>
  <p><strong>Aggregation.</strong> Per-strategy figures are signal-count-weighted means across groups, so a
  group contributing 900 signals counts nine times a group contributing 100. Group-level rows with fewer
  than three signals are excluded; degenerate one- and two-signal groups produce meaningless Sharpe values
  (one observed row reached &minus;1.1&times;10<sup>16</sup>) that would otherwise dominate any mean.</p>
  <p><strong>Evaluation.</strong> All performance is <em>shadow</em> performance: every signal every strategy
  emits is scored independently against the price series, with no capital constraint, no position limit,
  and no competition between strategies for allocation. This measures signal quality, not portfolio
  outcome, and the two are not interchangeable.</p>
  <p><strong>One exit rule, all three regimes.</strong> A signal is resolved by walking forward one bar at a
  time from its entry: the opening gap is checked before the intrabar range, and the stop before the target,
  so a bar that could have hit both is booked as a loss. A bar where neither triggers means the position is
  still open, and the walk continues. A signal whose stop and target are both still untouched when the data
  runs out is <strong>excluded</strong> rather than closed at an arbitrary price. The regimes differ in one
  respect only, and deliberately: oracle and base enter at the next bar&rsquo;s open, while naive enters at
  the close of its own withheld forecast bar, because that bar must have closed before a naive trade can
  exist at all.</p>
  <p>Excluding unresolved signals is a mild survivorship selection &mdash; a trade that drifts to the end of
  the window without touching either barrier is dropped rather than counted as the roughly flat outcome it
  was. Force-closing it at the last bar was the alternative, and it reintroduces exactly the arbitrary exit
  this rule removes. One consequence to expect: oracle and base signal counts are lower here than in the
  previous edition, because those regimes previously force-closed every signal and therefore counted every
  signal.</p>
</section>

<section>
  <h2 class="sec"><span class="no">04</span>Results</h2>
  {figure()}
  {summary_table()}

  <h3>4.1 &mdash; Without prediction, the corpus does not work</h3>
  <p>The naive regime is the weakest of the three. Across {BOX["naive"]["n"]:,} signals, <strong>{BOX["naive"]["pos"]} of {NSTRAT}
  strategies</strong> hold a positive signal-weighted Sharpe, the median strategy sits at
  <strong>{BOX["naive"]["med"]:+.2f}</strong>, mean win rate at {BOX["naive"]["win"]:.1f}%, and the median return per trade is negative at
  <strong>{BOX["naive"]["pnl"]:+.3f}%</strong>. More than three-quarters of the corpus loses money with no forward
  information, and the third quartile does not reach zero at all ({BOX["naive"]["q3"]:+.2f}).</p>
  <p>The survivors are mostly not forecast-shaped. The strongest, <code>bollinger_validation</code>, posts
  +6.54 &mdash; but on 145 signals, a fraction of a percent of the volume the mainstream strategies
  generate. The two thickest positives, <code>support_confluence</code> (+1.29) and
  <code>martingale_floor</code> (+0.98), are structural rather than directional: both derive their edge from
  bracket geometry and win-rate asymmetry rather than from calling direction. Strip out the forecast and
  what remains profitable is largely what never depended on direction in the first place.</p>
  <p>This includes implementations of families that have been standard teaching material in quantitative
  finance for decades &mdash; RSI filters, trend following, ATR bracketing, range trading, Bollinger
  validation, gap trading. In this corpus, given no forward-looking information, they are marginal at best
  and unprofitable in the majority.</p>

  <h3>4.2 &mdash; Prediction is the difference, but it does not rescue everything</h3>
  <p>Handed a perfect one-step forecast, the corpus moves decisively. The median Sharpe rises to
  <strong>{BOX["oracle"]["med"]:+.2f}</strong>, mean win rate to <strong>{BOX["oracle"]["win"]:.1f}%</strong>, and median return per trade to
  <strong>{BOX["oracle"]["pnl"]:+.3f}%</strong> &mdash; roughly five times the base model&rsquo;s. The oracle beats the naive
  floor on <strong>30 of {NSTRAT} strategies</strong>. The upper tail is where the effect is most visible:
  oracle&rsquo;s third quartile is <strong>{BOX["oracle"]["q3"]:+.2f}</strong> against naive&rsquo;s {BOX["naive"]["q3"]:+.2f}, and its best performer,
  <code>cross_asset_rank</code>, reaches {BOX["oracle"]["mx"]:+.2f}. Whatever these strategies are for, forecast quality is
  the input that makes them work.</p>
  <p>And yet <strong>{NSTRAT - BOX["oracle"]["pos"]} of {NSTRAT} strategies remain unprofitable with perfect foresight.</strong>
  Oracle&rsquo;s worst result, <code>volume_fade</code> at &minus;27.44, is far worse than its own naive
  result of &minus;0.18 &mdash; perfect information made it decisively worse, because a strategy that acts
  confidently on a correct forecast in the wrong direction loses faster than one that barely acts at all.
  Oracle&rsquo;s distribution is by far the widest of the three in both directions, spanning {BOX["oracle"]["mn"]:+.2f} to
  {BOX["oracle"]["mx"]:+.2f}. This is the finding that resists a comfortable reading: for a third of the corpus, no
  improvement in forecasting will help, because the decision rule sitting on top of the forecast is itself
  broken.</p>

  <h3>4.3 &mdash; The base model supplies real information</h3>
  <p>The pretrained Kronos model, with no fine-tuning, moves the corpus decisively on breadth. It lifts
  <strong>{BOX["base"]["pos"]} of {NSTRAT} strategies</strong> above zero &mdash; more than twice the naive count of
  {BOX["naive"]["pos"]}, and within two of the oracle&rsquo;s own {BOX["oracle"]["pos"]} &mdash; and beats the naive floor head-to-head on
  <strong>28 of {NSTRAT}</strong>. The median Sharpe turns positive at
  <strong>{BOX["base"]["med"]:+.2f}</strong> and median return per trade at <strong>{BOX["base"]["pnl"]:+.3f}%</strong>, double the
  naive figure&rsquo;s magnitude and opposite in sign.</p>
  <p>The individual movements are large where they matter. <code>open_gap</code> goes from &minus;0.72 to
  +2.55; <code>high_low</code> from &minus;0.83 to +0.94; <code>conditional_path</code> from &minus;1.04 to
  +0.65; <code>rsi_divergence</code> from &minus;1.42 to +0.15. These are strategies that do not work at all
  without a forecast and do work with this one. The model is not decorative.</p>
  <p>One detail is worth naming because it looks like a contradiction. The base regime&rsquo;s mean win rate
  is <strong>{BOX["base"]["win"]:.1f}%</strong>, <em>below</em> naive&rsquo;s {BOX["naive"]["win"]:.1f}%, while its median return per trade is
  positive and naive&rsquo;s is not. The model is not winning more often; it is winning better. What it
  changes is which side of an asymmetric bracket the trade sits on, which is what a directional forecast
  should change and what a win-rate count is blind to.</p>

  <div class="callout">
    <h4>The qualification this result carries</h4>
    <p>The base model nearly matches the oracle on <em>breadth</em> and captures little of its
    <em>magnitude</em>. It puts {BOX["base"]["pos"]} strategies above zero against perfect foresight&rsquo;s {BOX["oracle"]["pos"]}, but
    its median is {BOX["base"]["med"]:+.2f} against oracle&rsquo;s {BOX["oracle"]["med"]:+.2f}, its third quartile {BOX["base"]["q3"]:+.2f} against
    {BOX["oracle"]["q3"]:+.2f} &mdash; a sixteenfold gap &mdash; and its best strategy reaches {BOX["base"]["mx"]:+.2f} against
    {BOX["oracle"]["mx"]:+.2f}. The model finds the direction; it does not yet find the conviction.</p>
    <p>A second measurement sharpens this. Ranking the {NSTRAT} strategies by Sharpe within each regime and
    correlating those rankings, the base ordering still resembles the <em>naive</em> ordering
    (Spearman <span class="mono">&rho;&nbsp;=&nbsp;+0.556</span>) more closely than it resembles the
    <em>oracle</em> ordering (<span class="mono">&rho;&nbsp;=&nbsp;+0.346</span>). The base model shifts
    performance levels upward across the board, but which strategies it favours is still governed
    substantially by the same structural factors that operate with no forecast at all. Reading this result
    as &ldquo;the base model approaches perfect foresight&rdquo; would be wrong; it clears the floor
    convincingly and has barely started up the wall.</p>
  </div>

  <h3>4.4 &mdash; What the full corpus says, where it can</h3>
  <p>The {NGROUPS}-group matched sample is small and non-random, so the two complete regimes deserve a reading of
  their own. Oracle and naive both cover all <strong>961 groups</strong> and share <strong>46 strategies</strong>, on
  2,726,372 and 2,713,432 signals respectively &mdash; a paired comparison over the whole corpus with no
  base-coverage constraint at all.</p>
  <p>There, the naive floor sits at a median of <strong>&minus;0.54</strong> with 11 of 46 strategies
  positive, and the oracle ceiling at <strong>+1.23</strong> with 25 of 46, the oracle beating the floor on 28
  of 46. Every directional conclusion in &sect;4.1 and &sect;4.2 survives at full corpus scale; the magnitudes
  are smaller.</p>
  <p>That gap is the sample-ordering effect, and it is worth measuring rather than asserting. The oracle
  median is {BOX["oracle"]["med"]:+.2f} on the {NGROUPS} matched groups against +1.23 on all 961 &mdash; the matched groups are
  materially easier for a perfect forecast, exactly as an ordering by oracle Sharpe predicts. Read the base
  column against the oracle column <em>within</em> the matched sample, where both faced the same assets. Do
  not read either against the full-corpus numbers above.</p>
</section>

<section>
  <h2 class="sec"><span class="no">05</span>Discussion</h2>
  <p>The three regimes describe an interval, and each strategy sits somewhere inside it. The interval&rsquo;s
  width is the honest measure of how much a strategy&rsquo;s reported performance is attributable to
  forecasting rather than to itself &mdash; and for this corpus, that width is most of the number.</p>
  <p>Three groups fall out of the paired comparison. <strong>Forecast-dependent strategies</strong>
  (<code>open_gap</code>, <code>high_low</code>, <code>conditional_path</code>, <code>rsi_divergence</code>)
  are unprofitable at the floor and profitable with the base model. These are the corpus&rsquo;s real
  assets, and their performance is a direct claim about forecast quality.
  <strong>Structurally profitable strategies</strong> (<code>support_confluence</code>,
  <code>martingale_floor</code>, <code>bollinger_validation</code>, <code>fade_extreme</code>) clear zero even
  at the floor; their edge is bracket geometry and does not depend on knowing direction.
  <strong>Irreparable strategies</strong> (<code>volume_fade</code>, <code>gbm_direction</code>,
  <code>pca_residual_reversal</code>, <code>atr_bracket</code>) lose money in all three regimes &mdash; nine
  of {NSTRAT} do &mdash; several of them losing <em>more</em> as information improves.</p>
  <p>That third group is the most actionable output here. A strategy that loses money with perfect
  foresight cannot be fixed by a better model, more training data, or a longer fine-tune. It should be
  repaired at the decision rule or removed from the corpus, and the resources currently spent evaluating
  it across every sweep should go elsewhere. A third of the corpus is in this position.</p>
  <p>For the strategies that are not, the gap between the base column and the oracle column is the
  remaining headroom, and it is large. The pretrained model has demonstrated that the coupling works. What
  it has not yet demonstrated is that the coupling is tight.</p>
</section>

<section>
  <h2 class="sec"><span class="no">06</span>Limitations</h2>
  <p>These results are internally consistent, but several constraints bound how far they generalise. They
  are listed in descending order of how much they should temper the conclusions.</p>
  <ol>
    <li><strong>Thin, non-random base coverage.</strong> The base sweep has re-run {NGROUPS} of 961 groups under the
    current exit rule, so the three-regime matched sample is {NGROUPS} groups against the previous
    edition&rsquo;s 451, and it is ordered by oracle Sharpe rather than shuffled. &sect;4.4 measures what that
    ordering is worth: the oracle median is {BOX["oracle"]["med"]:+.2f} on these {NGROUPS} groups against +1.23 on all 961.
    Every three-regime figure in this note should be read as provisional and expected to move as coverage
    grows. The two-regime figures in &sect;4.4 are not affected &mdash; oracle and naive are complete. This is
    a transient state, not a design choice, and it is the single largest caveat in this edition.</li>
    <li><strong>Superseded: the exit rule is now identical across regimes.</strong> Earlier editions carried
    this as the study&rsquo;s largest caveat &mdash; oracle and base were scored by an evaluator that checked
    stop and target one bar ahead and then closed at that bar&rsquo;s close, while naive walked forward to a
    genuine trigger, so the floor and the ceiling were not measured with the same ruler. As of 29 August 2026
    all three regimes share one walk-forward rule (&sect;3.4) and every figure in this edition comes from a
    re-sweep under it. What replaces the caveat is narrower: unresolved signals are now excluded rather than
    force-closed, which is a mild survivorship selection the naive regime always had and the other two have
    now acquired.</li>
    <li><strong>No transaction costs.</strong> No commission, spread, slippage, borrow, or financing is applied
    anywhere. Median base-regime return per trade is {BOX["base"]["pnl"]:+.3f}%, which is inside realistic round-trip
    costs for much of this universe. The base regime&rsquo;s profitability claim is a claim about signal
    content, not about a tradeable edge.</li>
    <li><strong>Shadow accounting is not a portfolio.</strong> Every signal is scored independently with equal
    notional and no capital constraint. Strategies do not compete for allocation, positions do not
    interact, and correlated signals are counted as if independently exploitable. That last point bites
    harder here than it would elsewhere: group members are selected <em>for</em> correlation (&sect;3.2), so
    simultaneous signals across a group are non-independent by construction, and the effective sample size
    is smaller than the raw signal counts suggest. Real deployment on constrained capital will not
    reproduce these figures.</li>
    <li><strong>No significance testing.</strong> {NSTRAT} strategies were compared across three regimes with no
    correction for multiple comparisons. With that many candidates, several positive results are expected by
    chance alone. Individual per-strategy figures &mdash; particularly thin ones like
    <code>bollinger_validation</code>&rsquo;s 145 naive signals &mdash; should be read as indicative, not
    established. The aggregate distributional claims rest on far firmer ground than any single row.</li>
    <li><strong>One interval, one window, one regime of markets.</strong> All results are daily bars over a single
    six-month period. There is no walk-forward across market regimes, no intraday replication, and no
    out-of-sample holdout period. Robustness across time is untested.</li>
    <li><strong>The corpus is not the literature.</strong> Section 4.1 reports that implementations of
    long-taught strategy families underperform without a forecast. This is a statement about
    <em>these implementations</em> under <em>this evaluator</em>, not a refutation of the underlying published
    work. A canonical strategy poorly specified will fail for reasons that have nothing to do with the
    concept.</li>
    <li><strong>The corpus contains fewer distinct strategies than it appears to.</strong> Eight names
    resolve to one strategy (&sect;3.3) and are collapsed here. Four further pairs coincide on most but not
    all groups &mdash; <code>expected_value</code>/<code>vol_target_sizer</code>,
    <code>range_trading</code>/<code>rqa_determinism</code>,
    <code>dynamic_bracket</code>/<code>inverse_variance</code>,
    <code>amount_flow</code>/<code>predicted_vwap</code> &mdash; and are <em>not</em> collapsed, because they
    are not exact aliases. If they turn out to be trivially different, the effective corpus is smaller than
    {NSTRAT} and every count here is correspondingly inflated. On this sample <code>atr_bracket</code>,
    <code>trend_following</code> and <code>implementation_shortfall</code> also agree to two decimals in the
    naive and base regimes, which is worth a look.</li>
    <li><strong>Previously reported failures did not recur.</strong> Earlier editions excluded two groups that
    raised <code>ValueError: cannot convert float NaN to integer</code> under no-prediction mode. The
    re-swept oracle and naive sweeps each completed all 961 groups with no failures, so nothing is excluded
    from those two regimes on this ground. The base sweep is still running and has not yet covered the whole
    list.</li>
  </ol>
</section>

<section>
  <h2 class="sec"><span class="no">07</span>Full results</h2>
  <p>All {NSTRAT} distinct strategies present in every regime, ordered by base-model Sharpe. Read each row left to right
  as floor, system, ceiling.</p>
</section>
</div>

<div class="wide">
  {full_table()}
</div>

<div class="wrap">
<section>
  <h2 class="sec"><span class="no">08</span>Data provenance</h2>
  <p>Every figure in this note is derived from a single SQLite database in the Kairos repository. The
  regimes share one table by design &mdash; naive and oracle results are distinguished by a
  <code>stage</code> column rather than separate schemas.</p>
  <dl class="prov">
    <dt>Database</dt><dd>data/pipeline_results.db</dd>
    <dt>Universe screening</dt><dd>universe_screen   -- run_id, symbol, asset_class, bars, dollar_volume,
                     ann_vol, atr_pct, liquidity_note, passed, fail_reason
                     (1d run_id 759: 38,445 screened, 1,851 passed)</dd>
    <dt>Pairwise correlation</dt><dd>correlation_pairs -- run_id, symbol_a, symbol_b, asset_class,
                     full_corr, rolling_corr_median, overlap_bars
                     (run_id 760: 1,697,403 pairs)</dd>
    <dt>Candidate groups</dt><dd>suggested_groups  -- run_id, group_id, asset_class, symbols,
                     mean_intra_corr   (run_id 760: 13,472 groups)</dd>
    <dt>Naive and oracle results</dt><dd>oracle_results  (stage = 'naive' | 'oracle')</dd>
    <dt>Base and finetuned results</dt><dd>model_results   (stage = 'base' | 'finetuned')</dd>
    <dt>Shared columns</dt><dd>run_id, stage, strategy_name, sharpe, signal_count, win_rate,
avg_pnl_per_trade, assets, interval, backtest_period</dd>
    <dt>oracle_results only</dt><dd>version          -- short git commit hash, written per insert</dd>
    <dt>model_results only</dt><dd>model_path       -- NULL for base (stock NeoQuasar/Kronos-base)</dd>
    <dt>Run metadata</dt><dd>runs             -- one row per pipeline invocation</dd>
    <dt>CSV mirrors</dt><dd>results/{{naive,oracle}}_oracle_results_&lt;ts&gt;.csv
results/base_model_results_&lt;ts&gt;.csv</dd>
    <dt>Group source</dt><dd>select_deduped_groups(conn, correlation_run_id=760)  -&gt; 961 groups</dd>
    <dt>Exit-rule cutoff</dt><dd>runs.timestamp &gt;= '2026-08-30'   -- the one-exit-rule re-sweep
                     (oracle 961/961, naive 961/961, base {NGROUPS}/961)
                     earlier rows use the retired one-bar rule and are excluded</dd>
    <dt>Table source</dt><dd>make_paper_table.py  -&gt; paper_table.json  -&gt; build_paper.py
audit_paper.py       re-derives every figure from the DB independently</dd>
    <dt>Screening criteria</dt><dd>kairos_pipeline.py: passes_universe_screen()  -- bars/volume/ATR gates
                     liquidity_threshold()    -- $50M equity, $10M crypto, 0 FX
                     min_bars = KairosSettings.lookback = 300
                     (NB: passes_universe_screen's own min_bars=200 default is
                      for unit tests; run_stage_universe always passes 300)</dd>
    <dt>Grouping criteria</dt><dd>kairos_pipeline.py: compute_pair_correlation()  -- log-return Pearson
                     MIN_ABS_CORR = {{"crypto": 0.75, "default": 0.6}}
                     greedy_group_pairs(max_group_size=4)</dd>
    <dt>Sweep runner</dt><dd>scripts/run_oracle_dedup.py --stage oracle|naive --workers 8
scripts/run_base_priority.py                      (base)</dd>
  </dl>
  <p>The matched sample is the intersection of <code>(assets, interval, backtest_period)</code> across all
  three stages. Per-strategy aggregation is signal-count-weighted:</p>
  <pre class="sql">SELECT strategy_name,
       COUNT(DISTINCT run_id)                                  AS n_groups,
       SUM(signal_count)                                       AS total_signals,
       SUM(sharpe            * signal_count) * 1.0 / SUM(signal_count) AS w_sharpe,
       SUM(win_rate          * signal_count) * 1.0 / SUM(signal_count) AS w_win_rate,
       SUM(avg_pnl_per_trade * signal_count) * 1.0 / SUM(signal_count) AS w_avg_pnl
FROM   oracle_results          -- or model_results for base / finetuned
WHERE  stage = ?
  AND  signal_count &gt;= 3       -- excludes degenerate 1-2 signal groups
GROUP  BY strategy_name
ORDER  BY w_sharpe DESC;</pre>
  <h3>8.1 &mdash; Revision note</h3>
  <p><strong>31 August 2026 &mdash; one exit rule.</strong> Until 29 August 2026 the oracle and base regimes
  were scored by an evaluator that checked stop and target exactly one bar ahead and then closed the position
  at that bar&rsquo;s close, while the naive regime walked forward until a genuine stop or target triggered.
  The floor and the ceiling were measured with different rulers, and part of the gap between them was an
  artefact of that difference rather than of forecast quality. It was listed as this study&rsquo;s largest
  limitation from the first edition. All three regimes now share the single walk-forward rule described in
  &sect;3.4, the whole corpus was re-swept under it, and that limitation is retired.</p>
  <p>Every figure in this edition comes from the re-sweep; no pre-fix row is used anywhere. The oracle and
  naive sweeps are complete at 961 of 961 groups. The base sweep is not &mdash; it stands at {NGROUPS} groups and is
  still running &mdash; so the three-regime matched sample fell from 451 groups and 45 strategies to {NGROUPS} groups
  and {NSTRAT}, on {BOX["naive"]["n"] + BOX["base"]["n"] + BOX["oracle"]["n"]:,} signals against 2,814,025. The previous edition&rsquo;s
  headline figures were: naive median &minus;0.72 with 13 of 45 positive; base +0.16 with 29 of 45; oracle
  +2.20 with 27 of 45 and a third quartile of +9.96. Two conclusions changed. The oracle&rsquo;s advantage
  over the base model is larger than it appeared, because the retired rule truncated the oracle&rsquo;s
  winners along with its losers. And the base model no longer puts <em>more</em> strategies above zero than
  the oracle does &mdash; that claim rested on the old ruler and is withdrawn; it now puts slightly fewer
  ({BOX["base"]["pos"]} against {BOX["oracle"]["pos"]}). Everything else held its direction. Read the
  three-regime numbers as provisional until base coverage catches up.</p>
  <p><strong>Earlier revision.</strong> The first published version of this paper reported a matched sample of 343 groups and 51 strategies,
  with an oracle median of +0.32 and a base median of +0.28 &mdash; and concluded from those that the base
  model came within 0.04 of the perfect-foresight median. Two corrections were applied afterwards.</p>
  <p>First, the eight aliased names described in &sect;3.3 were counted as eight strategies rather than one.
  All eight are unprofitable in every regime and sit near the middle of the oracle distribution, so seven
  redundant copies dragged the oracle median down by roughly a factor of seven. Second, a targeted backfill
  of the naive sweep &mdash; 186 groups the oracle and base sweeps had covered and naive had not &mdash;
  grew the matched sample from 343 groups to 451.</p>
  <p>The direction of every conclusion survived both corrections; several magnitudes did not. The
  &ldquo;within 0.04 of perfect foresight&rdquo; claim was an artefact of the aliasing and is withdrawn: the
  corrected gap between the base and oracle medians is +0.16 against +2.20. The naive floor also rose
  (8 of 51 profitable, median &minus;0.76, becoming 13 of 45 and &minus;0.72), largely because the backfill
  added FX and commodity groups, which are the one segment that is profitable without any forecast at all.
  That last result is treated in the companion note, <em>Where Strategies Travel</em>.</p>
  <p>Reproducing a single regime for one group requires no database access:</p>
  <pre class="sql">uv run ./strategy/kairos_strategies.py --interval 1d --backtest_period 6m \\
    --pred_samples 100 --assets &lt;SYMBOLS&gt; --no_disabled_filter \\
    --no-prediction                 # oracle
    --no-prediction --naive-baseline # naive
    #                               # (omit both for base)</pre>
</section>

<section>
  <h2 class="sec"><span class="no">09</span>Conclusion</h2>
  <p>Separating the forecast from the rule that consumes it changes what a backtest number means. On this
  corpus, over {BOX["naive"]["n"] + BOX["base"]["n"] + BOX["oracle"]["n"]:,} signals matched across all three regimes &mdash; and 5.4 million more in the
  two-regime check of &sect;4.4 &mdash; the separation gives three results that hold together.</p>
  <p>Strategy logic alone does not carry the corpus. With no forward information, {NSTRAT - BOX["naive"]["pos"]} of {NSTRAT} strategies lose
  money and the median sits below zero; the survivors win on bracket geometry rather than on direction.
  Forecasting is the term that matters &mdash; perfect foresight lifts the median to {BOX["oracle"]["med"]:+.2f} and the upper
  quartile to {BOX["oracle"]["q3"]:+.2f} &mdash; but it is not a universal solvent: {NSTRAT - BOX["oracle"]["pos"]} of {NSTRAT} stay unprofitable knowing the
  future, and some strategies lose <em>more</em> the better their information gets. And the pretrained model
  earns its place, more than doubling the count of profitable strategies and coming within two of the count
  perfect foresight achieves, while capturing about a sixteenth of the oracle&rsquo;s upper-quartile
  magnitude and still ranking strategies more like the no-information regime than like the oracle.</p>
  <p>The practical reading: the coupling between model and corpus is real and worth investing in, roughly
  half the corpus should be repaired or retired regardless of model quality, and the distance between the
  base column and the oracle column is the space a fine-tuned model has to prove itself in. That last
  measurement is the next experiment.</p>
</section>
</div>

<footer>
  Kairos Project &middot; research note, {STAMP} &middot; generated from data/pipeline_results.db at commit {COMMIT}.<br>
  Matched sample: {NGROUPS} asset groups, {NSTRAT} distinct strategies, {BOX["naive"]["n"] + BOX["base"]["n"] + BOX["oracle"]["n"]:,} signals across three regimes;
  961 groups in the complete oracle&ndash;naive pair.
  Figures are shadow-performance measurements without transaction costs and are not investment advice.
</footer>
'''

# Two flavours of the same page:
#   docs/papers/  -> standalone file opened straight from disk, needs the doctype
#                    the Artifact publish pipeline would otherwise supply itself
#                    (without it browsers render in quirks mode).
#   scratchpad/   -> Artifact publish source, which must NOT carry doctype/head/body.
STANDALONE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">\n')

out = HERE / "prediction_premium.html"
out.write_text(STANDALONE_HEAD + HTML if HERE.name == "papers" else HTML)
print(f"wrote {out} ({len(HTML):,} bytes, {len(rows)} strategy rows)")

# Run from the scratchpad (the path the published Artifact is bound to), also
# refresh the repo copy. REPO_PAPERS is only a convenience for that direction;
# running this script from docs/papers/ needs nothing but HERE.
REPO_PAPERS = pathlib.Path("/media/baz/MonkeyWorks/PycharmProjects/Kairos/docs/papers")
if HERE.name != "papers" and REPO_PAPERS.is_dir():
    (REPO_PAPERS / "prediction_premium.html").write_text(STANDALONE_HEAD + HTML)
    print(f"wrote {REPO_PAPERS / 'prediction_premium.html'}")
