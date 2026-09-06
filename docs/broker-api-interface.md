# What Kairos needs from a broker/exchange API

Two capability tiers. Only the first is built.

## Tier 1 — cost & tradeability discovery (implemented)

Answers "what does this instrument cost to trade here, and can we trade it
at all" — the question the IBKR sweep (`docs/ibkr-cost-discovery.md`)
answered empirically and that turned out to matter more than execution
itself: IBKR's real answer (flat $1 commission floor, only 50/153 symbols
tradeable) is what ruled it out for Kairos's trade size.

The shared contract is `ExchangeProbe` in
[`scripts/exchange_probe.py`](../scripts/exchange_probe.py):

| Method | Purpose | IBKR's backing call | Typical ccxt-exchange backing call |
|---|---|---|---|
| `connect(args)` | Open a session | `IB().connect(host, port, clientId)` | `ccxt.<exchange>({'apiKey': ..., 'secret': ...})` (public data needs no key) |
| `disconnect()` | Close it | `ib.disconnect()` | usually a no-op / `exchange.close()` |
| `resolve_instrument(symbol) -> InstrumentMeta \| None` | Symbol → tradeable contract + tick/size metadata | `reqContractDetails` | `exchange.load_markets()[symbol]` |
| `get_reference_price(instrument) -> float \| None` | A price to size a probe order off, or just to record | `reqMktData` (delayed snapshot) | `exchange.fetch_ticker(symbol)['last']` |
| `get_cost_model(instrument, ref_price, base_currency) -> CostModel` | Commission (floor + rate) and margin, if any | two `whatIfOrder` dry-runs + `infer_commission_model` (Pattern A) | one `exchange.fetch_trading_fee(symbol)` call (Pattern B) |
| `get_fx_rate(currency, base) -> float \| None` | Normalize a foreign-currency margin/notional to account base currency | spot `Forex` quote via `reqMktData` | usually `1.0` — most crypto exchanges quote everything in one settlement currency; only relevant if `currency != base_currency` |

`InstrumentMeta` and `CostModel` (dataclasses in the same module) are the
row shape both patterns produce — see the module for exact fields. Margin
fields are `None` for spot/cash instruments; most crypto exchanges have no
margin concept at all, so that's the normal case, not a gap.

**Two cost-discovery patterns**, because they don't share enough real
structure to force into one mechanism:

- **Pattern A — dry-run/whatIf-order probing.** No API exposes the fee
  schedule directly; you infer floor + marginal rate from two probe orders
  at different notional. This is IBKR's whole mechanism —
  `scripts/ibkr_instruments.py`'s `sweep_one()` is the reference shape, and
  `infer_commission_model()` (in `exchange_probe.py`) is the shared piece
  of it.
- **Pattern B — direct fee-schedule lookup.** The exchange just tells you
  the maker/taker rate (ccxt's `fetch_trading_fee`, standardized across
  every ccxt-supported exchange). No probing, no inference — `get_cost_model`
  is one API call. `exchange_probe.sweep_simple()` drives this pattern
  end-to-end generically; see `FakeExchangeProbe` in
  `tests/unit/test_exchange_probe.py` for the minimal shape that proves it.

Three of the four crypto candidates from the broker survey (Bitvavo,
Kraken, Bybit EU) are Pattern B — see `docs/exchanges/*.md` for the
per-exchange mapping. IBKR is the only Pattern A example so far. **Finst
turned out to have no public API at all** (institutional-only, not in
ccxt) — neither pattern applies until that changes; see
`docs/exchanges/finst.md`. **Alpaca Europe doesn't fit either pattern**:
it's Broker-as-a-Service, not an account with an externally-set fee to
discover — see `docs/exchanges/alpaca-europe.md`.

## Tier 2 — live execution (not implemented)

From `roadmap/phase-5-auto-execution.md`'s original `kairos/broker.py`
sketch, gated on Phase 4 (paper trading matching backtest) and not started:

| Method | Purpose |
|---|---|
| `place_order(instrument, side, qty, order_type, ...)` | Submit a real order |
| `cancel(order_id)` | Cancel a working order |
| `get_positions()` | Current holdings |
| `get_balance()` | Account cash/margin state |

Nothing implements this yet — no exchange has a Kairos-owned account beyond
the IBKR paper account. Listed here for completeness so Tier 1 and Tier 2
aren't confused: everything built this session is Tier 1 (discovery),
which needs no funded account and (for the ccxt exchanges) often no
account at all, since market data and fee schedules are usually public.

## See also

- [`docs/playbooks/add-exchange-probe.md`](playbooks/add-exchange-probe.md)
  — the concrete steps to add a Tier 1 probe for a new exchange.
- [`docs/ibkr-cost-discovery.md`](ibkr-cost-discovery.md) — IBKR's own
  probe, API gotchas, and measured figures (Pattern A reference).
- [`docs/exchanges/`](exchanges/) — per-exchange connection notes for the
  4 crypto candidates + Alpaca Europe, written with this interface in mind.
