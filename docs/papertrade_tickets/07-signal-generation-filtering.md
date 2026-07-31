# Factor 7: Signal generation & strategy filtering

Source: `docs/papertrade_loss_analysis.md` §4, Factor 7

## Problem

`KairosOrchestrator._apply_meta_filters` (entropy > 3.0, bimodality/kurtosis
< -1.0), the per-strategy `KurtosisFilterStrategy`/`LiquidityFilterStrategy`
wrappers (`kurtosis_max=10.0`, `min_volume_percentile=10.0`), the ~18
permanently-disabled strategies plus per-profile disabled sets, and the
`min_ev_pct=0.10%` gate all determine which signals are even allowed to
reach allocation.

Factor 3 shows realized costs land right at 0.15% — so a signal only needs
`ev_pct > 0.10%` to be considered, a margin thinner than the realized cost
itself.

## Statistic to optimize

Per-strategy/per-asset EV-net-of-realistic-cost (not the theoretical 0.15%
assumption but the *measured* per-asset-class cost from Factor 3), and
oracle Sharpe as the strategy-level upper bound already used by
`resolve_disabled_strategies`.

## Concrete changes

- [ ] Raise `min_ev_pct` above 0.10% (e.g., closer to or above the
  empirically-measured 0.15% realized cost) so only signals that clear costs
  with real margin are taken — right now a signal can pass the gate with an
  EV edge *smaller* than what it will actually pay in costs.
- [ ] Re-run the oracle sweep behind `resolve_disabled_strategies` on recent
  data, specifically checking whether the top losing tickers from the
  recorded run (`LDO-USD`, `AAVE-USD`, `BTC-USD`, `FIL-USD`, `ATOM-USD`,
  `XTZ-USD`, `AXS-USD`) should have more strategies disabled for them
  specifically, rather than relying on the current interval/asset-class-level
  defaults.

## Files

- `strategy/kairos_orchestrator.py` (`_apply_meta_filters`, `resolve_disabled_strategies`)
- `strategy/kairos_execution.py` (`KurtosisFilterStrategy`, `LiquidityFilterStrategy`)
- `strategy/allocation.py` (`min_ev_pct` gate)

## Note

Directly informed by `03-transaction-cost-accrual.md`'s measured 0.15%
realized cost figure.
