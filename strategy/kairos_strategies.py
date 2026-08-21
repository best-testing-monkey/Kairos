from narwhals.dtypes import Unknown
import sys
from collections import Counter
from tabnanny import verbose

from pandas import DataFrame

from kairos.adapter import to_kronos_frame
from kairos.calendars import future_timestamps
import kairos_predcache

sys.path.insert(0, '.')

import argparse
import json
import sqlite3
import ta as ta_lib
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sqlitedict import SqliteDict

warnings.filterwarnings('ignore')

import torch
from model import Kronos, KronosTokenizer, KronosPredictor

sys.path.append("../")
import price_cache
from kairos.data import get_forecast_window, fetch_price_data_local_fallback
# model imports are deferred to _ensure_model_loaded() so --no-prediction
# runs never touch HuggingFace Hub or trigger its auth warning.
from typing import Callable, List, Optional, Sequence

import pandas as pd

from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig, print_results
from kairos_backtest import BARS_PER_DAY, bars_per_year, KairosSettings
from kairos.config import _state
from kairos_predcache import PredictionCache

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── Configuration ────────────────────────────────────────────────────────────
# Defaults live in KairosSettings; the __main__ block calls KairosSettings.configure(args).
SYMBOL = KairosSettings.symbol
LOOKBACK = KairosSettings.lookback
PRED_LEN = KairosSettings.pred_len
PRED_SAMPLES = KairosSettings.pred_samples
INITIAL_CAPITAL = KairosSettings.initial_capital
OUTPUT_DIR = KairosSettings.output_dir

bt_tokenizer :KronosTokenizer | None = None
bt_model = None
bt_predictor: None | KronosPredictor = None
_loaded_model_src = None  # (tokenizer_src, model_src) requested/resolved by the most recent
                          # _prepare_model_switch() call -- used for cache-clearing bookkeeping.
                          # NOT necessarily materialized into bt_predictor yet; see _weights_loaded_src.
_weights_loaded_src = None  # (tokenizer_src, model_src) actually materialized into bt_predictor, or None

# _shared_keys replaced the old per-process _prediction_cache dict
# (2026-08-11): rather than caching prediction DataFrames here too (on top
# of kairos_predcache's own disk/sqlite/mem layers -- see that module's
# docstring), this just remembers which shared-cache key predict_all_batch
# resolved for each symbol during its cache-lookup loop, so the later
# _dist_cache-population loop and predict_kairos_cloud don't need to
# recompute _shared_cache_key(). Much smaller footprint than caching actual
# prediction data a second time in-process.
_shared_keys: dict[str, str] = {}  # symbol -> cache key

_dist_cache: dict = {}  # (symbol, last_bar_ts) -> KairosDistribution
# NOTE: neither cache carries model identity. Whenever _ensure_model_loaded
# switches to a different (tokenizer_src, model_src) pair, both caches MUST
# be cleared or a two-pass (base -> finetuned) flow would silently reuse
# base-model predictions for the finetuned pass.

_DIST_CACHE_MAX_ENTRIES = 5000  # see _dist_cache_put's docstring

def _dist_cache_put(key, value) -> None:
    """Write to _dist_cache with a hard size cap, clearing the whole dict
    (not per-entry LRU eviction) on overflow.

    _dist_cache is only cleared on a model switch (_prepare_model_switch),
    same as _shared_keys -- correct for the base -> finetuned two-pass flow,
    but prewarm_prediction_cache's model-major base sweep processes many
    groups x dates against the SAME model without ever switching, so without
    this cap it grows for the whole sweep. Originally found 2026-08-08 as
    the actual dominant leak after fixes to the OTHER two overlapping
    prediction caches (kairos_predcache's in-memory LRU, and the
    now-removed _prediction_cache dict this file used to also keep) each
    only partially helped -- each KairosDistribution held here carries the
    full raw sample list *and* a concatenated DataFrame copy *and* a stats
    dict, making it the fattest of the (then-three, now-two) overlapping
    caches. Entries are a convenience mirror (recomputing
    distribution_for(preds) from an already-cached prediction is cheap), so
    clearing early costs a recompute, not correctness.
    """
    _dist_cache[key] = value
    if len(_dist_cache) > _DIST_CACHE_MAX_ENTRIES:
        _dist_cache.clear()

_no_data_fallback_warned: set = set()  # symbols we've already printed the

# ─────────────────────────────────────────────────────────────────────────────

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_data(symbol, lookback, pred_len):
    """Return context window + actual future bars for the backtest period."""
    raw = fetch_data_raw(lookback=lookback, pred_len=pred_len, symbol=symbol)

    # Set the context cut-off to pred_len bars before the last available bar so
    # that y_ts generated by get_forecast_window falls inside the historical
    # window and we can compare predictions against real prices.
    context_end = raw.index[-(pred_len + 1)]

    x_df, x_ts, y_ts = get_forecast_window(
        symbol=symbol,
        interval=KairosSettings.interval,
        lookback=lookback,
        pred_len=pred_len,
        end=context_end,
        amount="auto",
    )

    # Ground truth: the pred_len bars that follow context_end in raw data.
    actual = raw.iloc[-pred_len:]['close'].copy()
    actual.index = pd.to_datetime(actual.index)

    return x_df, x_ts, y_ts, actual


def is_24_7_crypto_symbol(symbol: str) -> bool:
    """True for symbols that trade around the clock, 7 days a week (no weekend padding needed).

    Crypto tickers from yfinance use the "-USD" suffix (e.g. BTC-USD, ETH-USD) and trade
    24/7/365. Everything else - equities/ETFs (plain tickers, e.g. SPY, QQQ, DIA, XLK),
    FX pairs ("=X" suffix, e.g. EURUSD=X), and futures ("=F" suffix) - is closed on
    weekends (and holidays), so a naive calendar-day window undershoots the real bar count.
    """
    return symbol.endswith("-USD")


EQUITY_TRADING_HOURS_PER_DAY = 6.5  # NYSE regular session, 9:30am-4:00pm ET


def is_limited_hours_equity_symbol(symbol: str) -> bool:
    """True for plain-ticker equities/ETFs (no -USD/=X/=F suffix).

    These trade only within a limited daily session (NYSE: 9:30am-4:00pm ET,
    6.5 hours) -- unlike FX/futures ("=X"/"=F" suffix), which trade nearly
    continuously through the weekday session, close enough to 24h/day that
    calendar_days_for_bars's weekend-only correction is accurate for them
    without a separate hours-per-day adjustment.
    """
    return not (symbol.endswith("-USD") or symbol.endswith(("=X", "=F")))


def calendar_days_for_bars(bars_needed: float, bars_per_day: float, symbol: str, buffer_days: int = 30) -> int:
    """Convert a bar count into a calendar-day window, padding for non-24/7 markets.

    Equities/ETFs and FX/futures trade roughly 5 out of every 7 days, so a naive
    bars_needed / bars_per_day calendar window undershoots real bar count by ~2/7.
    We multiply by 7/5 for those symbols (plus a small flat cushion) before adding
    the existing flat buffer_days, so the window still contains enough trading days.
    Crypto symbols (24/7) are left unpadded to preserve existing behavior.

    For limited-hours equities/ETFs specifically, the weekday-only correction
    alone is NOT enough at intraday granularity (bars_per_day > 1): `bars_per_day`
    is computed from BARS_PER_DAY, which assumes 24 hours of trading per day
    (correct for crypto, a reasonable approximation for near-continuous
    FX/futures) -- but an equity trading day is only ~6.5 hours (NYSE), not 24.
    Confirmed live (2026-08-21): fetch_data_raw raised "Not enough data for
    CB: need 300 bars, got 252" under the weekday-only correction alone (52
    calendar days requested, ~37 trading days x ~7 bars/day =~ 259, undershooting
    300) -- every 1h equity signal in a live kairos_signals.run() sweep silently
    failed and got swallowed by run()'s per-group exception handling, so this
    went unnoticed until traced through a report file's Failures section.
    Rescale bars_per_day itself to the real ~6.5 bars/day for these symbols
    before applying the weekend padding.
    """
    if bars_per_day > 1 and is_limited_hours_equity_symbol(symbol):
        bars_per_day = bars_per_day * (EQUITY_TRADING_HOURS_PER_DAY / 24.0)
    raw_days = bars_needed / bars_per_day
    if not is_24_7_crypto_symbol(symbol):
        raw_days = raw_days * (7 / 5) + 5
    return int(raw_days) + buffer_days


def fetch_data_raw(symbol, lookback, pred_len=0, min_bars=None, as_of=None) -> DataFrame:
    price_cache.configure(remote=False)

    from datetime import date, timedelta
    interval = KairosSettings.interval
    bars_per_day = BARS_PER_DAY.get(interval, 1)
    # Yahoo Finance hard limits by interval (days of history available)
    yf_max_days = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
        "60m": 729, "90m": 60, "1h": 729,
    }.get(interval, 5 * 365)
    bars_needed = max(min_bars or 0, lookback + pred_len)
    days_needed = min(calendar_days_for_bars(bars_needed, bars_per_day, symbol), yf_max_days)

    end_dt = as_of.date() if as_of is not None else date.today()
    start_dt = end_dt - timedelta(days=days_needed)
    end_str, start_str = end_dt.isoformat(), start_dt.isoformat()

    raw = price_cache.get_price_data(symbol, start_date=start_str, end_date=end_str, interval=interval)
    if raw is None or raw.empty:
        # price_cache marks a whole ticker as no-data in its no_data_tickers
        # table after a single failed fetch (e.g. a keyless provider), which
        # can hide data that is already sitting in the local prices table --
        # observed in production: a finetune_next backtest crashed here for a
        # symbol whose *training* fetch, moments earlier in the same run, had
        # already succeeded via this exact fallback. Try it before giving up.
        raw = fetch_price_data_local_fallback(symbol, pd.Timestamp(start_dt), pd.Timestamp(end_dt), interval, price_cache.DB_PATH)
        if raw is not None and not raw.empty:
            if symbol not in _no_data_fallback_warned:
                print(f"  [{symbol}] price_cache returned None; using direct local SQLite fallback")
            _no_data_fallback_warned.add(symbol)
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {symbol}")

    raw = raw.sort_index().copy()
    raw.columns = [c.lower() for c in raw.columns]
    idx = pd.to_datetime(raw.index)
    raw.index = idx.tz_convert(None) if idx.tz is not None else idx

    if as_of is not None:
        # Round down to the nearest bar: drop anything timestamped after
        # the simulated "now", so intraday intervals (h12/h6/1h/30m/15m/5m)
        # never leak a bar the caller couldn't actually have seen yet.
        raw = raw[raw.index <= as_of]

    if len(raw) < lookback + pred_len:
        raise RuntimeError(
            f"Not enough data for {symbol}: need {lookback + pred_len} bars, got {len(raw)}"
        )
    return raw


# ── Prediction ────────────────────────────────────────────────────────────────

def _model_switch_needed(requested_src, loaded_src) -> bool:
    """Pure decision: does the requested (tokenizer_src, model_src) pair
    require a (re)load?

    True when nothing is loaded yet (`loaded_src is None`) or when the
    requested pair differs from what's currently loaded. Kept as a pure
    function (no globals, no I/O) so it's unit-testable without touching
    the actual model-loading path.
    """
    return loaded_src is None or requested_src != loaded_src


def _prepare_model_switch(model_path=None, tokenizer_path=None):
    """Cheap, pure bookkeeping: resolve the requested (tokenizer_src,
    model_src) pair and, if it differs from what's currently "loaded"
    (`_loaded_model_src`), clear `_dist_cache`/`_shared_keys` and update
    `_loaded_model_src`.

    Does NOT touch disk/HF/GPU -- no from_pretrained, no KronosPredictor
    construction. Safe to call speculatively (e.g. before a cache lookup)
    without paying any model-load cost. Actually materializing the weights
    into `bt_predictor` is `_materialize_model`'s job.

    Returns the resolved (tok_src, mdl_src) tuple.
    """
    global _loaded_model_src

    tok_src = tokenizer_path or "NeoQuasar/Kronos-Tokenizer-base"
    mdl_src = model_path or "NeoQuasar/Kronos-base"
    requested_src = (tok_src, mdl_src)

    if _model_switch_needed(requested_src, _loaded_model_src):
        _dist_cache.clear()
        _shared_keys.clear()
        _loaded_model_src = requested_src

    return requested_src


def _materialize_model(requested_src):
    """Ensure `bt_predictor` actually holds the weights for `requested_src`.

    No-op if `bt_predictor` is already materialized for this exact src
    tuple (`_weights_loaded_src == requested_src`). Otherwise unloads
    whatever's currently loaded (if any) and does the heavy from_pretrained
    / GPU-move / TF32-or-INT8-quant / KronosPredictor construction.
    """
    global bt_tokenizer, bt_model, bt_predictor, _weights_loaded_src

    if bt_predictor is not None and _weights_loaded_src == requested_src:
        return

    tok_src, mdl_src = requested_src

    if bt_predictor is not None:
        # Switching to a different model: unload the old one first.
        print(f"Switching Kronos model: {_weights_loaded_src} -> {requested_src}")
        old_model, old_tokenizer, old_predictor = bt_model, bt_tokenizer, bt_predictor
        bt_model = bt_tokenizer = bt_predictor = None
        del old_model, old_tokenizer, old_predictor
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # gc.collect() frees the Python-level tensor/module objects, but glibc's
        # malloc doesn't return those freed arena pages back to the OS on its
        # own -- RSS keeps a high-water mark per switch instead of shrinking,
        # which live papertrade runs showed as a ~1-1.3GB step at EVERY model
        # switch (not per-date -- confirmed 2026-08-13 via memory_monitor_heap's
        # growth checkpoints landing exactly at switches, flat within a group's
        # 183-date sweep), eventually exhausting the 6-8GB budget once enough
        # finetuned groups had been swept. malloc_trim(0) explicitly asks glibc
        # to release free contiguous pages back to the OS; Linux/glibc only
        # (matches this module's other Linux-only assumptions, e.g. GPU recovery).
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass

    print(f"Loading Kronos model from {mdl_src} ...")
    bt_tokenizer = KronosTokenizer.from_pretrained(tok_src)
    bt_model = Kronos.from_pretrained(mdl_src)

    from kairos_gpu import ensure_cuda
    has_cuda = ensure_cuda()

    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  → GPU mode: autocast FP16, TF32 matmuls enabled")
    else:
        import torch.nn as nn
        bt_model = torch.quantization.quantize_dynamic(
            bt_model, {nn.Linear}, dtype=torch.qint8
        )
        print(f"  → CPU mode: INT8 dynamic quantisation, {torch.get_num_threads()} threads")

    bt_predictor = KronosPredictor(bt_model, bt_tokenizer, max_context=512)
    _weights_loaded_src = requested_src


def _ensure_model_loaded(model_path=None, tokenizer_path=None):
    """Thin wrapper preserving the original single-call behavior: resolve +
    prepare the requested src (cache-clear bookkeeping), then materialize it
    into bt_predictor if not already loaded. Used directly by run_model()
    and any other caller that always needs weights ready immediately."""
    requested_src = _prepare_model_switch(model_path, tokenizer_path)
    _materialize_model(requested_src)


def run_model(x_df, x_ts, y_ts, pred_len, sample_count=1,
              model_path=None, tokenizer_path=None, return_samples=False):
    _ensure_model_loaded(model_path, tokenizer_path)
    assert bt_predictor != None, "predictor is None!!"

    return bt_predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=sample_count,
        verbose=False,
        return_samples=return_samples,
    )


def _model_checkpoint_fingerprint(model_path) -> str:
    """Cheap identity fingerprint for a model checkpoint, for cache-key use.

    Local finetuned checkpoints (model_path is a directory) can be retrained
    in place at the same path -- fingerprint off model.safetensors' size +
    mtime so a retrain busts stale cache entries instead of silently serving
    them. HF repo ids (e.g. the base model) aren't local paths and don't get
    retrained in place, so there's nothing to stat -- the id string alone is
    already a stable identity.
    """
    if not model_path:
        return ""
    try:
        if not os.path.isdir(model_path):
            return ""
    except (TypeError, ValueError):
        return ""

    weights_path = os.path.join(model_path, "model.safetensors")
    try:
        st = os.stat(weights_path)
        return f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        pass

    # Defensive fallback (not expected in practice): newest mtime across
    # all files directly in the directory. Empty/unreadable directory -> "".
    try:
        newest = None
        with os.scandir(model_path) as it:
            for entry in it:
                try:
                    if not entry.is_file():
                        continue
                    mtime_ns = entry.stat().st_mtime_ns
                except OSError:
                    continue
                if newest is None or mtime_ns > newest:
                    newest = mtime_ns
        return str(newest) if newest is not None else ""
    except OSError:
        return ""


def _shared_cache_key(symbol, df, mdl_src, pred_len):
    """Build the shared-disk `kairos_predcache` key for one (symbol, df,
    model, pred_len) combination.

    Extracted from predict_all_batch's per-symbol loop so both that
    function's real cache-lookup path and is_batch_cached's read-only
    precheck build byte-identical keys off a single source of truth.
    """

    lookback_for_hash = min(KairosSettings.lookback, len(df))
    content_hash = kairos_predcache.content_hash_for_closes(
        df["close"].iloc[-lookback_for_hash:]
    )
    fingerprint = _model_checkpoint_fingerprint(mdl_src)
    return kairos_predcache.make_key(
        symbol=symbol,
        interval=KairosSettings.interval,
        bar_timestamp=df.index[-1],
        lookback_len=lookback_for_hash,
        pred_samples=KairosSettings.pred_samples,
        model_id=mdl_src,
        content_hash=content_hash,
        pred_len=pred_len,
        checkpoint_fingerprint=fingerprint,
    )


def is_batch_cached(assets: dict, model_path=None, pred_len=1) -> bool:
    """Read-only precheck: True iff every symbol in `assets` already has a
    shared-disk `kairos_predcache` hit for the resolved model id, i.e. a
    predict_all_batch(assets, model_path=model_path) call would trigger no
    real model load.

    False if the shared cache is inactive (kairos_predcache.get_cache()
    returns None -- KAIROS_PRED_CACHE_DIR unset) or if any symbol is a
    miss. Never loads a model, never writes to the shared cache --
    purely a lookup.

    Uses PredictionCache.has(), NOT .get(): .get() always deserializes a
    disk hit and promotes it into the in-memory LRU as a side effect, which
    for a pure existence check was quietly filling that LRU on every
    prewarm check-pass call (see PredictionCache.has's docstring for the
    2026-07-29 leak this caused).
    """
    import kairos_predcache

    shared_cache = kairos_predcache.get_cache(0)
    if shared_cache is None:
        return False

    mdl_src = model_path or "NeoQuasar/Kronos-base"
    for symbol, df in assets.items():
        key = _shared_cache_key(symbol, df, mdl_src, pred_len)
        if not shared_cache.has(key):
            return False
    return True


def predict_all_batch(assets: dict, model_path=None, tokenizer_path=None, build_distributions: bool = True) -> dict:
    """Predict all assets in one batched GPU call instead of N sequential calls.

    model_path / tokenizer_path: optional HF repo id or local path forwarded to
    _prepare_model_switch()/_materialize_model(). Passing a different
    model_path than what's currently loaded triggers an in-process model
    swap (see _model_switch_needed) -- but only if at least one requested
    symbol isn't already satisfied by the shared kairos_predcache (see that
    module's docstring for its sqlite/disk/mem layers): _materialize_model()
    (the actual from_pretrained / GPU-move work) is deferred until after the
    cache-lookup loop below, and skipped entirely when every symbol is a
    cache hit. There's no per-process prediction cache here anymore (the old
    _prediction_cache dict was removed 2026-08-11) -- every lookup and write
    goes through the shared cache directly; `_shared_keys` below just
    remembers the resolved key per symbol for the _dist_cache-population
    loop and predict_kairos_cloud to reuse.

    build_distributions=False (used by kairos_papertrade.py's
    prewarm_prediction_cache(), which discards this function's return value
    entirely -- its only purpose is the shared-cache put() side effect
    above) skips building AssetPrediction/KairosDistribution and writing to
    _dist_cache. Root-caused 2026-08-13: _dist_cache only clears on a model
    switch (once per finetuned group's ~183-date sweep), so a full group's
    worth of KairosDistribution objects -- each holding the full raw sample
    list *and* a concatenated DataFrame copy *and* a stats dict, per
    _dist_cache_put's docstring -- accumulated every prewarm sweep even
    though prewarm never reads a single one, matching the ~1GB-per-group RSS
    growth observed live across several runs. Returns {} in this mode.
    """
    from kairos_meta import AssetPrediction, KairosDistribution
    import kairos_predcache

    pred_len = 1
    requested_src = _prepare_model_switch(model_path, tokenizer_path)

    shared_cache = kairos_predcache.get_cache(0)

    df_list, x_ts_list, y_ts_list = [], [], []
    cached_results = {}
    uncached_symbols = []

    for symbol, df in assets.items():
        # cache_key = (symbol, df.index[-1])
        shared_key = _shared_cache_key(symbol, df, requested_src[1], pred_len)
        _shared_keys[symbol] = shared_key

        if shared_cache is not None:
            shared_hit = shared_cache.get(shared_key)
            if shared_hit is not None:
                cached_results[symbol] = shared_hit
                continue

        lookback = min(KairosSettings.lookback, len(df))
        x_df, x_ts = to_kronos_frame(df, lookback, amount="auto")
        y_ts = future_timestamps(x_ts.iloc[-1], KairosSettings.interval, 1, _state.calendar, _state.tz)
        df_list.append(x_df)
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)
        uncached_symbols.append(symbol)

    if uncached_symbols:
        _materialize_model(requested_src)
        assert bt_predictor != None, "predictor is None!!"

        seq_lens = [x.shape[0] for x in df_list]
        return_samples = KairosSettings.pred_samples > 1
        if len(set(seq_lens)) == 1:
            pred_lists = bt_predictor.predict_batch(
                df_list, x_ts_list, y_ts_list,
                pred_len=pred_len, sample_count=KairosSettings.pred_samples, return_samples=return_samples, verbose=False
            )
        else:
            pred_lists = [
                bt_predictor.predict(df, x_ts, y_ts, pred_len=pred_len, sample_count=KairosSettings.pred_samples,
                                     return_samples=return_samples, verbose=False)
                for df, x_ts, y_ts in zip(df_list, x_ts_list, y_ts_list)
            ]
        for symbol, preds in zip(uncached_symbols, pred_lists):
            if not isinstance(preds, list):
                preds = [preds]
            cached_results[symbol] = preds
            if shared_cache is not None and symbol in _shared_keys:
                shared_cache.put(_shared_keys[symbol], preds)
    
    if not build_distributions:
        return {}

    result = {}
    for symbol, preds in cached_results.items():
        dist_key = (symbol, assets[symbol].index[-1])
        dist = _dist_cache.get(dist_key)
        if dist is None:
            from kairos_backtest import distribution_for
            dist = distribution_for(preds)
            _dist_cache_put(dist_key, dist)
        result[symbol] = AssetPrediction(
            symbol=symbol,
            dist=dist,
            current_price=float(assets[symbol]["close"].iloc[-1]),
            history=assets[symbol],
        )
    return result


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest(predicted_close: pd.Series, actual_close: pd.Series,
             initial_capital: float, threshold: float):
    """
    Signal rule: if predicted daily return > threshold → buy;
                 if predicted daily return < -threshold → sell / short.
    P&L is computed on actual close prices.
    """
    pred = predicted_close.reset_index(drop=True)
    n = min(len(pred), len(actual_close))
    price = actual_close.iloc[:n].reset_index(drop=True)

    capital = initial_capital
    position = 0  # shares held (negative = short)
    trades = []
    equity = []

    for i in range(n):
        p = float(price.iloc[i])
        if pd.isna(p):
            equity.append(capital + position * (price.dropna().iloc[0] if position else 0))
            continue

        # Signal from predicted return
        if i < len(pred) - 1:
            pred_ret = (float(pred.iloc[i + 1]) - float(pred.iloc[i])) / float(pred.iloc[i])
        else:
            pred_ret = 0.0

        target_pos = 1 if pred_ret > threshold else (-1 if pred_ret < -threshold else 0)

        if target_pos != np.sign(position):
            # Close existing position
            if position != 0:
                capital += position * p
                trades.append(dict(day=i, action='CLOSE', price=p,
                                   shares=position, capital=capital))
                position = 0
            # Open new position
            if target_pos != 0:
                shares = int(capital / p) * target_pos
                position = shares
                capital -= shares * p
                trades.append(dict(day=i, action='BUY' if target_pos > 0 else 'SHORT',
                                   price=p, shares=shares, capital=capital))

        portfolio_val = capital + position * p
        equity.append(portfolio_val)

    # Close any open position at end
    if position != 0 and n > 0:
        last_p = float(price.iloc[n - 1])
        capital += position * last_p
        trades.append(dict(day=n - 1, action='CLOSE', price=last_p,
                           shares=position, capital=capital))

    equity_series = pd.Series(equity, index=actual_close.index[:n])
    return equity_series, trades


def compute_metrics(equity: pd.Series, initial_capital: float, trades: list, interval: str = "1d"):
    rets = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] - initial_capital) / initial_capital
    bpy = bars_per_year(interval)
    annual_ret = (1 + total_ret) ** (bpy / max(len(rets), 1)) - 1
    vol = rets.std() * np.sqrt(bpy)
    sharpe = (annual_ret - 0.03) / vol if vol > 0 else 0.0
    peak = equity.expanding().max()
    max_dd = ((equity - peak) / peak).min()
    trade_pairs = [(t['price'], trades[i + 1]['price'])
                   for i, t in enumerate(trades[:-1])
                   if t['action'] in ('BUY', 'SHORT') and
                   trades[i + 1]['action'] == 'CLOSE']
    trade_rets = [(s - b) / b for b, s in trade_pairs]
    win_rate = (sum(1 for r in trade_rets if r > 0) / len(trade_rets)
                if trade_rets else 0.0)
    return dict(total_return=total_ret, annual_return=annual_ret,
                volatility=vol, sharpe=sharpe, max_drawdown=max_dd,
                win_rate=win_rate, trades=len(trades),
                final_capital=equity.iloc[-1])


# ── Signal parsing ────────────────────────────────────────────────────────────

def parse_signals_config(signals_str):
    """Parse '--signals' CLI string into a config dict."""
    if not signals_str or signals_str.lower() == 'none':
        return {}
    config: dict[str, list[int] | list[Unknown] | list[int]| dict[str, int|float]  | None] = {
        'sma_periods': [],
        'ema_periods': [],
        'bb': None,
        'rsi': None,
        'stoch': None,
        'macd': None,
    }
    for token in signals_str.split(','):
        t = token.strip().upper()
        if t.startswith('SMA_'):
            config['sma_periods'].append(int(t[4:]))
        elif t.startswith('EMA_'):
            config['ema_periods'].append(int(t[4:]))
        elif t.startswith('BB_'):
            config['bb'] = {'period': int(t[3:]), 'std': 2.0}
        elif t == 'BB':
            config['bb'] = {'period': 20, 'std': 2.0}
        elif t == 'RSI':
            config['rsi'] = {'period': 14}
        elif t == 'STOCH':
            config['stoch'] = {'k': 14, 'd': 3}
        elif t == 'MACD':
            config['macd'] = {'fast': 12, 'slow': 26, 'signal_period': 9}
    return config


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(df, config):
    """
    Compute technical signals from a date-indexed OHLCV DataFrame using pandas_ta.
    Pass context + backtest period so rolling windows warm up correctly.
    Returns dict of signal_name -> pd.Series (same index as df).
    """
    if not config:
        return {}

    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close
    out = {}

    for period in config.get('sma_periods', []):
        out[f'SMA_{period}'] = ta_lib.trend.SMAIndicator(close, window=period).sma_indicator()

    for period in config.get('ema_periods', []):
        out[f'EMA_{period}'] = ta_lib.trend.EMAIndicator(close, window=period).ema_indicator()

    bb = config.get('bb')
    if bb:
        bb_ind = ta_lib.volatility.BollingerBands(close, window=bb['period'], window_dev=bb['std'])
        out['BB_upper'] = bb_ind.bollinger_hband()
        out['BB_lower'] = bb_ind.bollinger_lband()
        out['BB_mid'] = bb_ind.bollinger_mavg()

    rsi_cfg = config.get('rsi')
    if rsi_cfg:
        out['RSI'] = ta_lib.momentum.RSIIndicator(close, window=rsi_cfg['period']).rsi()

    stoch = config.get('stoch')
    if stoch:
        stoch_ind = ta_lib.momentum.StochasticOscillator(
            high, low, close, window=stoch['k'], smooth_window=stoch['d']
        )
        out['STOCH_K'] = stoch_ind.stoch()
        out['STOCH_D'] = stoch_ind.stoch_signal()

    macd = config.get('macd')
    if macd:
        macd_ind = ta_lib.trend.MACD(
            close, window_slow=macd['slow'], window_fast=macd['fast'], window_sign=macd['signal_period']
        )
        out['MACD_line'] = macd_ind.macd()
        out['MACD_signal'] = macd_ind.macd_signal()
        out['MACD_hist'] = macd_ind.macd_diff()

    return out

# ── Interactive control panel ─────────────────────────────────────────────────

def predict_kairos_cloud(signal: Optional[pd.DataFrame]= None, **kwargs) -> List[pd.DataFrame]:
    """Interactive/scripted single-symbol prediction entry point.

    Unlike predict_all_batch (the batched path used by the papertrade/signal
    generation loops), this asserts the shared kairos_predcache is active
    (KAIROS_PRED_CACHE_DIR must be set) rather than tolerating it being off --
    there's no per-process fallback cache to fall back to since
    _prediction_cache was removed (2026-08-11). `signal is None` triggers the
    interactive "fetch fresh data and describe the run" path; otherwise it
    checks the shared cache for `_shared_keys[symbol]` (populated by a prior
    predict_all_batch call for this symbol) before predicting.
    """
    shared_cache: Optional[PredictionCache] = kairos_predcache.get_cache(0)
    assert shared_cache is not None, "kairos_predcache.get_cache() failed to supply an object"

    pred_historic = kwargs.get('pred_historic', 0)
    pred_num = kwargs.get('pred_num', 1)
    model_path = kwargs.get("model") or KairosSettings.model or "NeoQuasar/Kronos-base"
    tokenizer_path = kwargs.get("tokenizer") or KairosSettings.tokenizer or "NeoQuasar/Kronos-Tokenizer-base"
    symbol = kwargs.get("symbol") or KairosSettings.symbol
    lookback = KairosSettings.lookback
    pred_samples = KairosSettings.pred_samples

    shared_key = _shared_keys[symbol]
    if signal is None:
        print("Kairos prediction cloud")
        print(f"   Symbol:             {symbol}")
        print(f"   Context window:     {lookback} bars")
        print(f"   Samples per bar:    {pred_samples} x")
        print()
        print("Step 1: Fetching data ...")
        x_df, x_ts, y_ts, actual = fetch_data(symbol, lookback, pred_historic)
    else:
        if shared_cache.has(shared_key):
            # pyrefly: ignore [bad-return]
            return shared_cache.get(shared_key)
        lookback = min(KairosSettings.lookback, len(signal))
        x_df, x_ts = to_kronos_frame(signal, lookback, amount="auto")
        y_ts = future_timestamps(x_ts.iloc[-1], KairosSettings.interval, 1, _state.calendar, _state.tz)

    result_list = run_model(
        x_df, x_ts, y_ts[:1], pred_num,
        sample_count=pred_samples,
        model_path=model_path, tokenizer_path=tokenizer_path,
        return_samples=(pred_samples > 1),
    )
    if not isinstance(result_list, list):
        result_list = [result_list]

    if signal is not None and shared_key is not None and result_list is not None:
        shared_cache.put(shared_key , result_list)
    return result_list


def _parse_period(period: str) -> tuple:
    """Parse a human period string into (count, unit).

    Args:
        period: Period string, e.g., '6m', '1y', '3m', '2w', '10d'

    Returns:
        Tuple of (count: int, unit: str)

    Raises:
        ValueError: If period format is invalid
    """
    import re as _re
    m = _re.fullmatch(r"(\d+)(d|w|m|y)", period.strip().lower())
    if not m:
        raise ValueError(
            f"Unrecognised backtest_period {period!r}. Use e.g. '6m', '1y', '3m', '2w', '10d'."
        )
    return int(m.group(1)), m.group(2)


def _period_to_bars(period: str, interval: str) -> int:
    """Convert a human period string (e.g. '6m', '1y') to a bar count."""
    n, unit = _parse_period(period)
    cal_days = {"d": n, "w": n * 7, "m": n * 30, "y": n * 365}[unit]
    bars_per_day = BARS_PER_DAY.get(interval, 1)
    return max(1, int(cal_days * bars_per_day))


def _period_to_weeks(period: str) -> float:
    """Convert a human period string (e.g. '6m', '1y') to weeks.

    Uses 365.25 days per year and 7 days per week.
    Matches the period parsing from _period_to_bars.
    """
    n, unit = _parse_period(period)
    # Convert to calendar days: use 365.25 days/year, 30.4375 days/month (365.25/12)
    cal_days = {
        "d": n,
        "w": n * 7,
        "m": n * 365.25 / 12,
        "y": n * 365.25,
    }[unit]
    # Convert calendar days to weeks (7 days per week)
    return cal_days / 7


_COMMODITY_ETFS = {"GLD", "SLV", "USO", "UNG", "DBC", "PDBC", "CPER", "COPX", "GDX"}


def asset_class_for(assets) -> str:
    """Classify a group of tickers into crypto/fx/commodity/equity/mixed.

    Rules: crypto tickers end in "-USD", fx tickers end in "=X", commodity
    tickers end in "=F" or are one of the known commodity ETFs
    (GLD/SLV/USO/UNG/DBC/PDBC/CPER/COPX/GDX). Everything else is equity.
    A group's class is the majority class of its symbols; if no single
    class has a strict majority, the group is "mixed".
    """
    classes = []
    for sym in assets:
        s = sym.strip()
        if s.endswith("-USD"):
            classes.append("crypto")
        elif s.endswith("=X"):
            classes.append("fx")
        elif s.endswith("=F") or s in _COMMODITY_ETFS:
            classes.append("commodity")
        else:
            classes.append("equity")
    if not classes:
        return "mixed"
    counts = Counter(classes)
    top_class, top_count = counts.most_common(1)[0]
    if top_count * 2 > len(classes):
        return top_class
    return "mixed"


# Disabled strategies per (interval, asset_class) fallback, used when no
# oracle-tested DB profile (data/pipeline_results.db, disabled_strategies
# table, keyed on the exact (interval, sorted-assets) pair) exists for the
# (interval, assets) pair - see resolve_disabled_strategies() below.
#
# Derived from the 2026-07-05 oracle (--no-prediction) shadow sweep at
# interval=1d, backtest_period=3m, across 27 asset groups (7 crypto, 10
# equity, 3 fx, 7 commodity/mixed-commodity). Rule: a strategy is disabled
# for a class if EITHER
#   (a) it had negative sharpe in >=60% of that class's groups AND
#       >=10 total signals across the class (avoids disabling on noise), OR
#   (b) mean sharpe (degenerate |sharpe|>1000 outliers excluded) <= -5.0
#       with >=2 groups negative.
# Mean sharpe values in the comments are the capped/filtered means used to
# apply the rule above.
_DISABLED_BY_CLASS: dict = {
    ("1d", "crypto"): {
        "volume_fade",              # mean sharpe -153.74, neg 6/6
        "cross_asset_spread",       # mean sharpe  -93.02, neg 5/6
        "volume_confirmation",      # mean sharpe  -36.64, neg 6/6
        "funding_rate_prediction",  # mean sharpe  -27.70, neg 6/6
        "path_v_shape",             # mean sharpe  -11.07, neg 6/6
        "bsts_decomposition",       # mean sharpe   -9.70, neg 6/6
        "rsi_divergence",           # mean sharpe   -9.33, neg 4/7
        "particle_filter",          # mean sharpe   -8.89, neg 6/6
        "inverse_variance",         # mean sharpe   -8.57, neg 6/6
        "dynamic_bracket",          # mean sharpe   -8.42, neg 6/6
        "close_direction",          # mean sharpe   -8.14, neg 6/6
        "hmm_regime",               # mean sharpe   -7.33, neg 3/6
        "distribution_overlap",     # mean sharpe   -6.63, neg 6/6
        "conditional_path",         # mean sharpe   -6.02, neg 6/6
    },
    ("1d", "commodity"): {
        "volume_fade",              # mean sharpe  -70.09, neg 6/6
        "tail_asymmetry",           # mean sharpe  -21.15, neg 4/6
        "rsi_filter",               # mean sharpe  -15.10, neg 5/6
        "cross_asset_spread",       # mean sharpe  -12.16, neg 2/5
        "volume_confirmation",      # mean sharpe  -10.05, neg 6/7
        "funding_rate_prediction",  # mean sharpe   -8.51, neg 7/7
        "hmm_regime",               # mean sharpe   -6.43, neg 4/6
        "distribution_overlap",     # mean sharpe   -5.94, neg 6/7
    },
    ("1d", "equity"): {
        "volume_fade",              # mean sharpe  -48.73, neg 4/7
        "rsi_filter",               # mean sharpe  -12.45, neg 7/10
        "tail_asymmetry",           # mean sharpe  -11.35, neg 6/10
        "funding_rate_prediction",  # mean sharpe  -11.10, neg 10/10
        "volume_confirmation",      # mean sharpe   -5.84, neg 9/10
        "distribution_overlap",     # mean sharpe   -4.80, neg 9/10 (>=60%, >=10 sig rule)
        "hmm_regime",               # mean sharpe   -4.48, neg 9/10 (>=60%, >=10 sig rule)
        "skew",                     # mean sharpe   -4.04, neg 7/10 (>=60%, >=10 sig rule)
        "conditional_path",         # mean sharpe   -1.18, neg 7/10 (>=60%, >=10 sig rule)
    },
    ("1d", "fx"): {
        "tail_asymmetry",           # mean sharpe  -12.26, neg 3/3
        "cross_asset_spread",       # mean sharpe   -8.31, neg 2/3
        "hmm_regime",               # mean sharpe   -5.29, neg 3/3
        "funding_rate_prediction",  # mean sharpe   -3.75, neg 3/3 (>=60%, >=10 sig rule)
        "distribution_overlap",     # mean sharpe   -2.68, neg 2/3 (>=60%, >=10 sig rule)
        "conditional_path",         # mean sharpe   -0.90, neg 2/3 (>=60%, >=10 sig rule)
    },
}


# Repo-root / DB-path derivation, matching kairos_signals.py's pattern. Kept
# as module constants for readability, but resolve_disabled_strategies()
# recomputes the actual path used per-call (when db_path is None) rather than
# baking DEFAULT_DB_PATH in at import time, so tests can override db_path and
# a fresh clone (no data/ dir yet) doesn't fail at import.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")


def resolve_disabled_strategies(interval: str, assets, db_path: str = None) -> set:
    """Resolve the set of disabled strategy names for a run.

    Resolution order:
      1. If data/pipeline_results.db's oracle_results table has ANY row for
         the exact (interval, sorted-assets-key) profile - i.e. the profile
         has been oracle-tested - return the (possibly empty) set of
         strategy names found in that DB's disabled_strategies table for the
         same profile. An empty result here is meaningful ("tested and
         clean") and does NOT fall through to the class fallback.
      2. Otherwise (profile never oracle-tested, DB missing, or any sqlite3
         error), fall back to (interval, asset_class) in _DISABLED_BY_CLASS,
         where asset_class is computed via asset_class_for(assets).
      3. Empty set if neither matches.

    `db_path` defaults to None, resolving to DEFAULT_DB_PATH at call time
    (not baked in at import time) so tests can point at a temp DB. Never
    imports kairos_pipeline (would create an import cycle) - plain sqlite3
    only, opened/closed per call.

    Note: this assumes oracle_results.assets was written with a consistently
    sorted CSV for a given profile going forward (run_stage_oracle /
    refresh_disabled_strategies in kairos_pipeline.py normalize independently
    at insert time); legacy rows written by an unsorted direct CLI call are a
    pre-existing data-quality issue out of scope here.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    assets_key = ",".join(sorted(assets))
    cls = asset_class_for(assets)
    class_fallback = _DISABLED_BY_CLASS.get((interval, cls), set())

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        tested = conn.execute(
            "SELECT 1 FROM oracle_results WHERE interval=? AND assets=? LIMIT 1",
            (interval, assets_key),
        ).fetchone()
        if tested is None:
            return class_fallback
        rows = conn.execute(
            "SELECT strategy_name FROM disabled_strategies WHERE interval=? AND assets=?",
            (interval, assets_key),
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.Error:
        return class_fallback
    finally:
        if conn is not None:
            conn.close()


_DEFAULT_ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos walk-forward backtest - Strategies based on predictions")
    parser.add_argument("--model", metavar="PATH", default=None,
                        help="Local path to finetuned Kronos predictor (defaults to NeoQuasar/Kronos-base)")
    parser.add_argument("--tokenizer", metavar="PATH", default=None,
                        help="Local path to Kronos tokenizer (defaults to NeoQuasar/Kronos-Tokenizer-base)")
    parser.add_argument("--symbol", metavar="SYM", default=SYMBOL,
                        help=f"Trading symbol (default {SYMBOL})")
    parser.add_argument("--interval", metavar="INTERVAL", default=None, dest="interval",
                        help='Bar size: "1d", "1h", "15m", etc. (default: 1d)')
    parser.add_argument("--assets", nargs="+", metavar="SYM", default=None, dest="assets",
                        help=f"Asset tickers to backtest (default: {' '.join(_DEFAULT_ASSETS)})")
    parser.add_argument("--backtest_period", metavar="PERIOD", default=None, dest="backtest_period",
                        help='Backtest duration: "6m", "1y", "3m", "2w", "10d", etc. (default: 6m)')
    parser.add_argument("--lookback", metavar="N", default=LOOKBACK, type=int,
                        help=f"Context window bars (default {LOOKBACK})")
    parser.add_argument("--pred_samples", metavar="N", default=PRED_SAMPLES, type=int,
                        help=f"Samples per bar (default {PRED_SAMPLES})")
    parser.add_argument("--initial_capital", metavar="N", default=INITIAL_CAPITAL, type=float,
                        help=f"Initial capital (default {INITIAL_CAPITAL})")
    parser.add_argument("--no-prediction", dest="no_prediction", action="store_true", default=False,
                        help="Replace model predictions with actual next-bar OHLCV (oracle baseline)")
    parser.add_argument("--export_json", metavar="PATH", default=None, dest="export_json",
                        help="Additionally dump summary/strategy_rankings/shadow_performance to this JSON path")
    parser.add_argument("--no_disabled_filter", dest="no_disabled_filter", action="store_true", default=False,
                        help="Bypass DB/class disabled-strategy resolution; evaluate every strategy "
                             "(used by the oracle pipeline stage)")

    args = parser.parse_args()
    KairosSettings.configure(args)

    assets = KairosSettings.assets or _DEFAULT_ASSETS
    if args.no_disabled_filter:
        disabled = set()
        print(f"  [info] --no_disabled_filter set - bypassing disabled-strategy resolution, all strategies enabled.")
    else:
        disabled = resolve_disabled_strategies(KairosSettings.interval, assets)
        if not disabled:
            profile_key = (KairosSettings.interval, ",".join(sorted(assets)))
            print(f"  [info] No oracle-tested disabled-strategy profile or class fallback for {profile_key} - all strategies enabled.")

    DEMO_BACKTEST_OVER_N_BARS = _period_to_bars(KairosSettings.backtest_period, KairosSettings.interval)

    config = OrchestratorConfig.for_interval(
        KairosSettings.interval,
        initial_capital=KairosSettings.initial_capital,
        cross_asset_ranking=True,
        online_weighting=True,
        partial_exits=True,
        max_horizon=3,
        no_prediction=KairosSettings.no_prediction,
        disabled_strategies=disabled,
    )

    orchestrator = KairosOrchestrator(
        predict_fn=predict_kairos_cloud,
        assets=assets,
        config=config,
        batch_predict_fn=predict_all_batch,
        model=KairosSettings.model,
        tokenizer=KairosSettings.tokenizer,
        symbol=KairosSettings.symbol,
    )

    lookback = KairosSettings.lookback
    results = orchestrator.run_backtest({
        sym: fetch_data_raw(sym, lookback, min_bars=lookback + DEMO_BACKTEST_OVER_N_BARS)
                 .tail(lookback + DEMO_BACKTEST_OVER_N_BARS)
        for sym in assets
    }, lookback=lookback)

    top_results = orchestrator.backtest_top_strategies(results, n=len(results["strategy_rankings"]))
    print_results(results, top_results)

    if args.export_json:
        # Additive-only: minimal JSON dump for the pipeline's oracle/model stages.
        # Does not alter stdout output or any behavior when the flag is absent.
        def _jsonable(obj):
            if isinstance(obj, dict):
                return {k: _jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_jsonable(v) for v in obj]
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        export_payload = {
            "summary": _jsonable(results.get("summary", {})),
            "strategy_rankings": _jsonable(results.get("strategy_rankings", [])),
            "shadow_performance": _jsonable(results.get("shadow_performance", {})),
            "strategy_build_stats": _jsonable(results.get("strategy_build_stats", {})),
            "signal_firing_count": _jsonable(results.get("signal_firing_count", 0)),
        }
        with open(args.export_json, "w") as _f:
            json.dump(export_payload, _f, indent=2)
        print(f"  [export_json] wrote {args.export_json}")
