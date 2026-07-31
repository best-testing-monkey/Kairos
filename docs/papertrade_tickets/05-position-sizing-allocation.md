# Factor 5: Position sizing / capital allocation

Source: `docs/papertrade_loss_analysis.md` §4, Factor 5

## Problem

`strategy/allocation.py`'s Kelly-shrinkage sizing (`n0=100`, `min_n=50`,
`kelly_mult=0.35`) combined with `max_pos_pct=15`, `top_k=3` (from
`--top-n`), and `gross_cap_pct=100` determines exactly how big each position
is. Given the measured 24.3% win rate / ~4.6x payoff ratio, the *true*
Kelly-optimal fraction is thin and highly sensitive to `p`/`b` estimation
error — oversizing here is what turns Factor 2's uncapped concurrency into
large realized drawdown.

## Statistic to optimize

The realized Kelly edge (`p_shrunk`, `kelly_frac`) vs. actual out-of-sample
win rate/payoff — and, jointly with Factor 2, the **peak portfolio-level**
(not just per-position) Kelly fraction actually deployed.

## Concrete changes

- [ ] Lower `kelly_mult` (currently 0.35) to reduce variance, especially
  while Factor 2 (uncapped concurrent exposure) is unaddressed — the two
  compound.
- [ ] Consider sizing at the **portfolio** level (aggregate Kelly across all
  currently open + newly selected positions) rather than
  per-position-in-isolation, since that's what actually drives drawdown.
- [ ] Re-validate `min_n=50`/`n0=100` shrinkage constants against how much
  real sample data (`n`) is typically available for the traded tickers —
  thin-sample signals get heavily shrunk toward `p=0.5`, but if `n` is
  systematically inflated or deflated for certain asset classes, sizing will
  be systematically biased too.

## Files

- `strategy/allocation.py` (Kelly shrinkage: `n0`, `min_n`, `kelly_mult`,
  `max_pos_pct`, `gross_cap_pct`)

## Note

Compounds with `02-portfolio-exposure-cap.md` — both address the same
underlying drawdown-amplification mechanism from different angles.
