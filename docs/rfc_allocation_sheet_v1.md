# RFC: Portfolio Allocation Sheet

Status: Draft
Scope: Signal report generator -- adds one sheet to the XLSX and ODS outputs and one section to the Markdown output.

## 1. Problem

The report currently emits per-signal advice with a per-signal "liquidity %" that is computed in isolation. Summed across ~153 signals/day the total is a multiple of 100%, so it cannot be executed as-is. The report gives no answer to the two questions that actually matter at execution time:

1. Which signals to take (selection).
2. How much equity per selected signal (sizing).

This RFC adds a deterministic selection + sizing layer as a derived view over the existing signal list. It does not change signal generation.

## 2. Non-goals

- No changes to strategy logic or backtesting.
- No live-position awareness (open exposure, freed capital slots). That requires state the report generator does not have; see Open Questions.
- No correlation-matrix clustering in v1 (static cluster map instead; see Open Questions).

## 3. Inputs

The signal fetch method is extended to return all fields as structured data. The prose advice string remains for human display only; nothing downstream parses it. (This resolves Issue #3.)

Required schema per signal (fetch method return / new columns in the signal table):

| Field | Type | Example | Notes |
|---|---|---|---|
| strategy | str | path_execution | |
| ticker | str | REMX | |
| direction | enum long/short | short | |
| entry | float | 79.73 | reference/entry price the SL/TP percentages are relative to |
| stop | float | 84.51 | |
| target | float | 73.71 | |
| ev_pct | float | 4.04 | empirical, from realized backtest exits |
| base_win_rate | float | 0.47 | |
| n | int | 161 | trades in backtest |
| backtest_period | str/dates | | already available per the source data |
| sharpe | float | | already available per the source data |
| advised_liquidity_pct | float | 11.0 | carried for traceability, IGNORED (see 4.1) |
| avg_win_pct | float | | REQUESTED, see Issue #1; nullable in v1 |
| avg_loss_pct | float | | REQUESTED, see Issue #1; nullable in v1 |
| avg_holding_days | float | | REQUESTED, see Issue #4; nullable in v1 |

With `entry` provided directly, the derived percentages become:

```
risk_pct   = abs(stop - entry)   / entry * 100
reward_pct = abs(target - entry) / entry * 100
```

Validation instead of parsing: reject a row `SCHEMA_ERROR` if direction is inconsistent with stop/target placement (long requires stop < entry < target; short requires target < entry < stop), or if any required field is missing/non-finite.

Migration note: for replaying old report files that only contain the prose string, a fallback regex parser may be kept in a separate module, but it is not part of the production path and its failures never block a report.

### 3.1 Config (new)

| Parameter | Default | Meaning |
|---|---|---|
| n0 | 100 | Shrinkage constant; weight of the "no edge" prior |
| min_n | 50 | Reject signals with fewer backtest trades |
| round_trip_cost_pct | 0.15 | Assumed total cost per round trip (spread + commission + slippage). Placeholder, calibrate per instrument class later |
| kelly_mult | 0.35 | Fractional Kelly multiplier |
| top_k | 12 | Max number of positions |
| max_pos_pct | 15 | Cap per position, % of equity |
| max_cluster_pct | 25 | Cap per correlation cluster, % of equity |
| gross_cap_pct | 100 | Total gross exposure cap |
| cluster_map | file | ticker -> cluster name, static mapping |

Defaults are deliberately round. They should be swept in Phantom Ledger against the historical signal stream and then re-rounded; do not ship precise-looking fitted values.

## 4. Computation

All values computed by the generator in Python and written as static values. No spreadsheet formulas. Rationale: (a) formula dialect differences between Excel and ODF are a maintenance tax, (b) the selection logic (clustering, top-K, iterative caps) is not cleanly expressible in cell formulas anyway, (c) the sheet is a report, not a calculator.

### 4.1 Ignore upstream liquidity %

The advised liquidity % is single-signal Kelly with no portfolio context (Issue #2). It is carried through for traceability and shown grayed out, but plays no role in selection or sizing.

### 4.2 Per-row derived columns

```
risk_pct    = abs(stop - entry) / entry * 100    # loss if stopped, % of entry
reward_pct  = abs(target - entry) / entry * 100  # gain at target, % of entry

# Payoff ratio: empirical when available, geometry as fallback
if avg_win_pct and avg_loss_pct:
    b        = avg_win_pct / avg_loss_pct
    loss_pct = avg_loss_pct                      # basis for score denominator
else:
    b        = reward_pct / risk_pct             # geometry-based approximation
    loss_pct = risk_pct

shrink      = n / (n + n0)                       # confidence weight in [0,1)
ev_shrunk   = ev_pct * shrink                    # empirical EV shrunk toward 0
ev_net      = ev_shrunk - round_trip_cost_pct    # after costs

p_shrunk    = 0.5 + (base_win_rate - 0.5) * shrink
kelly_raw   = p_shrunk - (1 - p_shrunk) / b      # binary Kelly on shrunk win rate
kelly_frac  = max(kelly_raw, 0) * kelly_mult

score       = ev_net / loss_pct                  # return per unit risk; ranking key
```

Notes:

- `ev_pct` is empirical (from realized backtest exits), while the geometry fallback for `b` assumes clean TP/SL fills. These disagree in practice; see 4.3 and Issue #1. Once avg_win_pct / avg_loss_pct are populated by the fetch method, the fallback path becomes dead code.
- If avg_holding_days is populated, an alternative ranking key `score_daily = ev_net / avg_holding_days` becomes available (Issue #4). Which key wins is a Phantom Ledger sweep question, not a design decision.

### 4.3 Data quality check: ev_implied vs ev_reported

```
ev_implied = base_win_rate * reward_pct - (1 - base_win_rate) * risk_pct
ev_ratio   = ev_pct / ev_implied     (guard ev_implied ~ 0)
```

On the sample data this diverges substantially -- see Issue #1 for the numbers and analysis. Flag `DATA_MISMATCH` (informational, not rejecting) when `ev_ratio` is outside [0.5, 2.0]. Rows carrying this flag have Kelly fractions of reduced trustworthiness. The check remains useful even after avg_win/avg_loss are populated: it then validates internal consistency of the backtest export itself (base_win_rate x avg_win - (1 - base_win_rate) x avg_loss should approximately reproduce ev_pct).

### 4.4 Selection algorithm

```
candidates = fetch_signals()   # structured, per section 3

# Gate
for c in candidates:
    reject(c, SCHEMA_ERROR)    if required field missing or SL/TP placement
                               inconsistent with direction
    reject(c, LOW_N)           if c.n < min_n
    reject(c, NEG_EV_NET)      if c.ev_net <= 0

# Collapse per asset
for ticker, group in group_by(candidates, ticker):
    if directions_disagree(group):
        reject(all of group, DIRECTION_CONFLICT)
    else:
        keep max(group, key=score); reject(rest, DUP_ASSET)

# Rank and take top K
survivors = sort(candidates, key=score, desc=True)
selected  = survivors[:top_k]
reject(survivors[top_k:], BELOW_TOPK)

# Size
for s in selected:
    s.alloc = min(s.kelly_frac, max_pos_pct)

# Cluster caps: proportionally scale down within any cluster over its cap
for cluster, group in group_by(selected, cluster):
    if sum(alloc) > max_cluster_pct:
        scale group allocations by max_cluster_pct / sum(alloc)
        annotate CLUSTER_CAPPED

# Gross cap: proportional scale-down if total > gross_cap_pct
if sum(all alloc) > gross_cap_pct:
    scale all by gross_cap_pct / sum(all alloc)

# Dust filter: drop positions whose final alloc is too small to beat costs
for s in selected:
    reject(s, DUST) if s.alloc < 1.0   # percent of equity; make configurable
```

Deterministic tie-break on equal score: higher n, then ticker alphabetical. Same input file must always produce the same output.

### 4.5 Rejection reason enum

`SCHEMA_ERROR, LOW_N, NEG_EV_NET, DIRECTION_CONFLICT, DUP_ASSET, BELOW_TOPK, DUST`
Informational flags (non-rejecting): `DATA_MISMATCH, CLUSTER_CAPPED, POS_CAPPED`

## 5. Sheet design (XLSX and ODS)

New sheet named `Allocation`, identical structure in both formats.

### 5.1 Layout

```
Row 1      Title: "Portfolio Allocation" + report date
Rows 3-12  Config block (two columns: parameter, value) incl. generator version
Row 14     Summary line: selected count, gross exposure %, cluster count
Row 16+    Section A: SELECTED table
(blank)    Section B: Cluster exposure table
(blank)    Section C: REJECTED table (compact)
```

### 5.2 Section A: selected positions

| Col | Header | Example (REMX row) |
|---|---|---|
| A | Ticker | REMX |
| B | Cluster | metals_miners |
| C | Strategy | path_execution |
| D | Dir | Short |
| E | Entry | 79.73 |
| F | Stop | 84.51 |
| G | Target | 73.71 |
| H | Risk % | 6.0 |
| I | Reward % | 7.6 |
| J | b | 1.27 |
| K | n | 161 |
| L | Win raw | 47.0% |
| M | Win shrunk | 48.2% |
| N | EV raw % | 4.04 |
| O | EV net % | 2.34 |
| P | Kelly raw | 7.2% |
| Q | Score | 0.39 |
| R | Alloc % | 2.5 |
| S | Alloc EUR | (Alloc % x equity, if equity configured, else blank) |
| T | Flags | DATA_MISMATCH |
| U | Advised liq % (ignored) | 11 |

Sorted by Alloc % descending. Column U grayed. Conditional formatting kept minimal: red text on Flags column when non-empty. No colors that do not survive ODS conversion.

### 5.3 Section B: cluster exposure

| Cluster | Positions | Gross % | Cap % | Capped? |
|---|---|---|---|---|
| energy_commodities | 1 | 13.9 | 25 | no |

### 5.4 Section C: rejected (compact)

One row per rejected signal: Ticker, Strategy, Dir, Score, Reason. Sorted by reason then score. This keeps the audit trail cheap: any "why is X not in the list" question is answered by the sheet itself.

## 6. Markdown section design

Appended after the existing signal list:

```markdown
## Portfolio Allocation

Config: n0=100 min_n=50 cost=0.15% kelly_mult=0.35 top_k=12
max_pos=15% max_cluster=25% gross_cap=100%

Selected 9 of 153 signals. Gross exposure: 61.4%.

| Ticker | Dir | Strategy | Entry | Stop | Target | EV net | Score | Alloc |
|--------|-----|----------|-------|------|--------|--------|-------|-------|
| NG=F   | Long | close_direction | 2.95 | 2.97 | 3.14 | 0.61% | 1.01 | 13.9% |
| ...    |      |                 |      |      |      |       |      |       |

Cluster exposure: energy_commodities 13.9%, healthcare 8.2%, ...

Rejected: 144 total -- DUP_ASSET 78, BELOW_TOPK 31, LOW_N 20,
NEG_EV_NET 12, DIRECTION_CONFLICT 3
```

Markdown shows selected + aggregate rejection counts only; the full rejected table lives in the spreadsheet. Keeps the markdown scannable.

## 7. Worked example (from the sample rows)

With defaults (n0=100, cost 0.15, kelly_mult 0.35):

| Ticker | risk | reward | b | n | shrink | EV net | Kelly frac | Score | Note |
|---|---|---|---|---|---|---|---|---|---|
| NG=F (close_direction) | 0.6 | 6.7 | 11.2 | 109 | 0.52 | 0.61 | 13.9% | 1.01 | wins NG=F collapse |
| NG=F (open_gap) | 1.0 | 5.4 | 5.4 | 79 | 0.44 | 0.56 | 12.1% | 0.56 | DUP_ASSET, loses to above |
| REMX | 6.0 | 7.6 | 1.27 | 161 | 0.62 | 2.34 | 2.5% | 0.39 | DATA_MISMATCH flag |
| V | 1.6 | 1.8 | 1.13 | 319 | 0.76 | 0.68 | 2.4% | 0.43 | high n, thin edge |
| CRM | 1.5 | 2.3 | 1.53 | 79 | 0.44 | 0.27 | 9.1% | 0.18 | low score despite decent Kelly |

Illustrates the intended behavior: raw EV ranking (REMX first) differs from score ranking (NG=F first) because score is per unit risk after shrinkage and costs. Also note Kelly and score can disagree (CRM); score decides selection, Kelly decides size.

## 8. Implementation notes

- Single module, e.g. `allocation.py`: `fetch_signals() -> [Candidate]` (structured, per section 3), `allocate(candidates, config) -> AllocationResult`, plus writers `write_xlsx_sheet`, `write_ods_sheet`, `write_md_section` consuming the same `AllocationResult`. Selection logic tested independently of rendering.
- Pure function of (candidates, config). No I/O inside `allocate`. Makes it directly reusable inside Phantom Ledger for the parameter sweep.
- Config in the existing config mechanism of the report generator; also echoed into the sheet and md output so every report is self-describing.
- Unit tests: golden-file test on the sample rows above; property test that output allocs always respect all caps; determinism test (same input twice -> byte-identical section); schema validation tests for each SCHEMA_ERROR condition.

## 9. Issues

Defects and shortcomings in the current system surfaced while writing this RFC. Numbered for cross-reference; severity is about impact on sizing correctness, not code quality.

**Issue #1 -- ev_pct contradicts the stated TP/SL exit model. Severity: high (bug).**
The advice text claims "Exit by TP/SL", but reported ev_pct is irreconcilable with base_win_rate x TP - (1 - base_win_rate) x SL. From the sample rows: REMX implied 0.39% vs reported 4.04% (off by ~10x, implying an average win of ~15% against a 7.6% TP); NG=F close_direction implied 2.32% vs reported 1.45% (off in the opposite direction). At least one of these is true: exits are not actually at TP/SL (time exits, trailing, gaps), the ev_pct calculation is wrong, or the TP/SL in the advice text is not the TP/SL the backtest used. Until root-caused, Kelly sizing from TP/SL geometry is built on numbers the backtest contradicts.
Fix: root-cause in the backtester; export avg_win_pct and avg_loss_pct per signal (schema in section 3) so sizing uses realized payoffs. The 4.3 consistency check then becomes a permanent regression guard.

**Issue #2 -- advised liquidity % is not executable. Severity: high (design flaw).**
Per-signal sizing ignores all other signals; daily totals reach multiples of 100% of equity. Anyone executing the column as printed is unknowingly leveraged severalfold.
Fix: this RFC. Consider removing or relabeling the column in the advice text once the Allocation sheet ships, so the unusable number stops looking authoritative.

**Issue #3 -- signal parameters embedded in prose. Severity: medium (data plumbing).**
Strategy, direction, ticker, SL/TP prices and percentages exist only inside a formatted sentence, forcing regex re-parsing of data the system already had structured. Any wording change silently breaks consumers.
Fix: structured fields returned by the fetch method / added as columns (section 3). Prose string demoted to display-only.

**Issue #4 -- no holding-time statistics. Severity: medium (opportunity cost invisible).**
EV per trade cannot be compared across signals with different horizons. A 3% EV signal holding for three weeks blocks capital that could cycle through many faster signals; the current data cannot express this.
Fix: export avg_holding_days (and ideally the exit-reason distribution: TP / SL / time) per signal; enables the score_daily ranking key in 4.2.

**Issue #5 -- flat cost assumption across futures, crypto, and equities. Severity: medium.**
0.15% round trip is simultaneously too high for liquid US equities and too low for crypto CFDs. Since ev_net gates selection, a wrong cost constant systematically mis-ranks asset classes.
Fix: cost table per asset class (or per ticker), same config mechanism.

**Issue #6 -- static cluster map understates correlation. Severity: low in v1, grows with position count.**
Hand-maintained ticker-to-cluster mapping misses cross-sector correlation (everything correlates in a drawdown) and rots as the universe changes.
Fix: pairwise return correlation from price_cache, cluster at rho > 0.6, refreshed monthly.

**Issue #7 -- allocation assumes a flat book. Severity: acceptable for a report, blocking for automation.**
The sheet sizes as if no positions are open. Real execution needs open-exposure state so new signals only fill freed capital.
Fix: out of scope here; belongs in Phantom Ledger consuming this sheet as input.

## 10. Open questions / v2

1. **Parameter sweep.** Replay the historical signal stream in Phantom Ledger over (top_k, n0, kelly_mult, cluster threshold, score vs score_daily); optimize geometric growth subject to max drawdown; validate on a held-out period; round the winning values before shipping them as defaults.
2. **Column vs fetch method.** Section 3 allows either adding structured columns to the signal table or extending the fetch method's return type. If other consumers read the table directly, columns are the safer contract; if the fetch method is the only access path, extending its return type avoids widening the table. Decide based on who else reads the table.
3. **Advice text after Issue #2 fix.** Whether the prose advice keeps showing a per-signal liquidity % at all, or replaces it with a pointer to the Allocation sheet.
