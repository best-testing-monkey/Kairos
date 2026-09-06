# Playbook: adding a new exchange cost-discovery probe

Adds a new exchange to the same measurement Kairos ran against IBKR
(`docs/ibkr-cost-discovery.md`): what does this instrument actually cost to
trade, and can we trade it at all. See
[`docs/broker-api-interface.md`](../broker-api-interface.md) for the
interface this implements and why it's split into two patterns.

Read `docs/exchanges/<name>.md` first if one exists for the exchange you're
adding — it already has the auth/endpoint mapping done.

## Which pattern applies

- **Fee schedule is a direct API call** (every ccxt-supported exchange —
  Bitvavo, Kraken, Bybit, and most others): **Pattern B**. Follow the steps
  below as written; you get the whole sweep loop for free from
  `exchange_probe.sweep_simple()`.
- **No fee-schedule endpoint, only dry-run order evaluation** (IBKR's
  situation): **Pattern A**. Reuse `exchange_probe.infer_commission_model`/
  `_num`/`upsert`/`connect_db`, but write your own sweep loop — copy the
  shape of `scripts/ibkr_instruments.py`'s `sweep_one()`, not `sweep_simple()`.
  Not covered step-by-step here since none of the current candidates need it.

## Steps (Pattern B)

1. **New file**: `scripts/<exchange>_instruments.py`. Copy this skeleton:

   ```python
   #!/usr/bin/env python
   """<Exchange> instrument + fee sweep. See docs/exchanges/<exchange>.md
   and docs/playbooks/add-exchange-probe.md."""
   import argparse
   import os
   import sys

   REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
   import exchange_probe as ep

   DB_PATH = os.path.join(REPO_ROOT, "data", "<exchange>_instruments.db")


   class <Exchange>Probe:
       broker = "<Exchange>"

       def __init__(self):
           self._client = None

       def connect(self, args):
           import ccxt
           self._client = ccxt.<exchange>({"enableRateLimit": True})
           self._client.load_markets()

       def disconnect(self):
           pass

       def resolve_instrument(self, symbol):
           m = self._client.markets.get(symbol)
           if not m:
               return None
           return ep.InstrumentMeta(
               native_symbol=m["id"], instrument_class="crypto",
               sec_type="spot", exchange=self.broker, currency=m["quote"],
               native_id=m["id"], min_tick=m["precision"].get("price"),
               size_increment=m["precision"].get("amount"),
               min_size=(m.get("limits", {}).get("amount", {}) or {}).get("min"),
           )

       def get_reference_price(self, instrument):
           t = self._client.fetch_ticker(instrument.native_symbol)
           return t.get("last")

       def get_cost_model(self, instrument, ref_price, base_currency):
           fee = self._client.fetch_trading_fee(instrument.native_symbol)
           taker = fee.get("taker")
           if taker is None:
               return ep.CostModel(status="no_commission")
           return ep.CostModel(status="ok", commission_min=0.0,
                                commission_rate_pct=taker * 100,
                                commission_currency=instrument.currency)

       def get_fx_rate(self, currency, base):
           return 1.0 if currency == base else None  # fill in if it matters

   def main():
       ap = argparse.ArgumentParser()
       ap.add_argument("--symbols", default="")
       ap.add_argument("--report", action="store_true")
       ap.add_argument("--force", action="store_true")
       ap.add_argument("--base-currency", default="EUR")
       args = ap.parse_args()

       conn = ep.connect_db(DB_PATH, ep.SCHEMA_TEMPLATE, "instruments")
       if args.report:
           ep.report(conn, "instruments")
           return

       probe = <Exchange>Probe()
       probe.connect(args)
       symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
       ep.sweep_simple(probe, conn, "instruments", symbols,
                        base_currency=args.base_currency, force=args.force)
       probe.disconnect()
       ep.report(conn, "instruments")


   if __name__ == "__main__":
       main()
   ```

2. **Fill in the three real methods** (`resolve_instrument`,
   `get_reference_price`, `get_cost_model`) against the exchange's actual
   SDK/ccxt surface. `docs/exchanges/<name>.md` names the exact calls.

3. **Universe/symbol list**: decide how `args.symbols` maps to the
   exchange's own symbol format (e.g. `BTC-USD` → ccxt's `BTC/USD`). Keep
   the mapping in this file, not in `exchange_probe.py` — it's the one
   genuinely exchange-specific piece `resolve_instrument` needs.

4. **Test the pure logic**, mirroring `tests/unit/test_ibkr_instruments.py`'s
   `TestYfToIb` class: pin the symbol-mapping function with a handful of
   cases (one per instrument class the exchange lists). You do **not** need
   to re-test `infer_commission_model`/`_num`/`upsert`/`sweep_simple` —
   those are already covered in `tests/unit/test_exchange_probe.py` and
   unchanged by adding an exchange.

5. **Run it**: `uv run --with ccxt scripts/<exchange>_instruments.py
   --symbols BTC-USD,ETH-USD` then `--report`. Public market data and fee
   schedules on ccxt exchanges usually need no API key — check
   `docs/exchanges/<name>.md`'s auth section before assuming you need one.

6. **Write the findings up**: a `docs/<exchange>-cost-discovery.md` if the
   sweep surfaces anything as load-bearing as IBKR's flat commission floor
   did — otherwise a short note in `docs/exchanges/<name>.md` is enough.

## What this buys you

Diff size for a new Pattern B exchange, if the mapping is straightforward:
one new file (~80-120 lines, mostly the three real methods), no changes to
`scripts/exchange_probe.py` or `scripts/ibkr_instruments.py`. Schema,
resumability, the upsert-rank guard, and reporting are already shared.

## What this does not cover yet

- **No live smoke test against a real sandbox/testnet.** This pass built
  the framework only — Baz's call, 2026-09-06: scaffolding + playbook now,
  real accounts and sandbox keys later (Bitvavo/Kraken/Finst/Bybit EU all
  currently accountless, same parked status DEGIRO/BUX have been in since
  2026-08-25).
- **Tier 2 (execution)** — `place_order`/`get_positions`/etc. — is
  undesigned; see `docs/broker-api-interface.md`.
