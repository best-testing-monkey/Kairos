"""Tests for kairos_strategies' model-switch-capable singleton loader.

No real model/GPU/network is touched: model.Kronos/KronosTokenizer/
KronosPredictor and kairos_gpu.ensure_cuda are monkeypatched with fakes.
_model_switch_needed is a pure function and is tested directly; the actual
load path (HF downloads, quantization) stays untested, matching the rest
of the codebase.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pandas as pd
import pytest

import kairos_strategies


class _FakeTokenizer:
    def __init__(self, src):
        self.src = src

    @classmethod
    def from_pretrained(cls, src):
        return cls(src)


class _FakeModel:
    def __init__(self, src):
        self.src = src

    @classmethod
    def from_pretrained(cls, src):
        return cls(src)


class _FakePredictor:
    def __init__(self, model, tokenizer, max_context=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context


@pytest.fixture(autouse=True)
def _reset_model_globals():
    """Every test starts from (and leaves behind) a clean singleton state,
    so tests don't leak a "loaded" model into other test modules."""
    def _clear():
        kairos_strategies.bt_tokenizer = None
        kairos_strategies.bt_model = None
        kairos_strategies.bt_predictor = None
        kairos_strategies._loaded_model_src = None
        kairos_strategies._weights_loaded_src = None
        kairos_strategies._prediction_cache.clear()
        kairos_strategies._dist_cache.clear()

    _clear()
    yield
    _clear()


def _patch_model_loading(monkeypatch, cuda_available=True):
    """Patch model.Kronos/KronosTokenizer/KronosPredictor and
    kairos_gpu.ensure_cuda so _ensure_model_loaded never touches a real
    model, HuggingFace Hub, or GPU/recovery ladder."""
    import model as model_module
    monkeypatch.setattr(model_module, "Kronos", _FakeModel, raising=False)
    monkeypatch.setattr(model_module, "KronosTokenizer", _FakeTokenizer, raising=False)
    monkeypatch.setattr(model_module, "KronosPredictor", _FakePredictor, raising=False)

    import kairos_gpu
    monkeypatch.setattr(kairos_gpu, "ensure_cuda", lambda *a, **kw: cuda_available)


# ============================================================================
# _model_switch_needed (pure function)
# ============================================================================

class TestModelSwitchNeeded:
    def test_nothing_loaded_needs_switch(self):
        assert kairos_strategies._model_switch_needed(("tok", "mdl"), None) is True

    def test_identical_pair_no_switch(self):
        assert kairos_strategies._model_switch_needed(("tok", "mdl"), ("tok", "mdl")) is False

    def test_different_model_src_needs_switch(self):
        assert kairos_strategies._model_switch_needed(("tok", "mdl2"), ("tok", "mdl")) is True

    def test_different_tokenizer_src_needs_switch(self):
        assert kairos_strategies._model_switch_needed(("tok2", "mdl"), ("tok", "mdl")) is True


# ============================================================================
# _ensure_model_loaded — switch-on-path-change + cache clearing
# ============================================================================

class TestEnsureModelLoadedSwitching:
    def test_first_load_sets_loaded_src_and_predictor(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a", tokenizer_path="repo/a-tok")

        assert kairos_strategies._loaded_model_src == ("repo/a-tok", "repo/a")
        assert kairos_strategies.bt_predictor is not None
        assert kairos_strategies.bt_predictor.model.src == "repo/a"

    def test_default_src_used_when_no_path_given(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded()

        assert kairos_strategies._loaded_model_src == (
            "NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base",
        )

    def test_second_call_same_path_is_a_noop(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")
        predictor_1 = kairos_strategies.bt_predictor

        kairos_strategies._ensure_model_loaded(model_path="repo/a")

        assert kairos_strategies.bt_predictor is predictor_1

    def test_switch_to_different_model_replaces_predictor(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")
        predictor_a = kairos_strategies.bt_predictor

        kairos_strategies._ensure_model_loaded(model_path="repo/b")
        predictor_b = kairos_strategies.bt_predictor

        assert predictor_b is not predictor_a
        assert predictor_b.model.src == "repo/b"
        assert kairos_strategies._loaded_model_src == (
            "NeoQuasar/Kronos-Tokenizer-base", "repo/b",
        )

    def test_switch_clears_prediction_and_dist_caches(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")

        # Seed the caches as if a prediction had already run against "a".
        kairos_strategies._prediction_cache[("BTC-USD", "t0")] = ["fake_pred"]
        kairos_strategies._dist_cache[("BTC-USD", "t0")] = "fake_dist"

        kairos_strategies._ensure_model_loaded(model_path="repo/b")

        assert kairos_strategies._prediction_cache == {}
        assert kairos_strategies._dist_cache == {}

    def test_same_model_reload_does_not_clear_caches(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")
        kairos_strategies._prediction_cache[("BTC-USD", "t0")] = ["fake_pred"]
        kairos_strategies._dist_cache[("BTC-USD", "t0")] = "fake_dist"

        kairos_strategies._ensure_model_loaded(model_path="repo/a")

        assert kairos_strategies._prediction_cache == {("BTC-USD", "t0"): ["fake_pred"]}
        assert kairos_strategies._dist_cache == {("BTC-USD", "t0"): "fake_dist"}

    def test_switch_calls_gc_collect(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")

        calls = {"n": 0}
        import gc as gc_module
        real_collect = gc_module.collect
        def _counting_collect(*a, **kw):
            calls["n"] += 1
            return real_collect()
        monkeypatch.setattr(gc_module, "collect", _counting_collect)

        kairos_strategies._ensure_model_loaded(model_path="repo/b")

        assert calls["n"] >= 1

    def test_switch_calls_cuda_empty_cache_when_cuda_available(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")

        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        calls = {"n": 0}
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1))

        kairos_strategies._ensure_model_loaded(model_path="repo/b")

        assert calls["n"] == 1

    def test_switch_skips_cuda_empty_cache_when_cuda_unavailable(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a")

        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        calls = {"n": 0}
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1))

        kairos_strategies._ensure_model_loaded(model_path="repo/b")

        assert calls["n"] == 0

    def test_tokenizer_path_change_alone_triggers_switch(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._ensure_model_loaded(model_path="repo/a", tokenizer_path="tok/1")
        predictor_1 = kairos_strategies.bt_predictor

        kairos_strategies._ensure_model_loaded(model_path="repo/a", tokenizer_path="tok/2")

        assert kairos_strategies.bt_predictor is not predictor_1
        assert kairos_strategies._loaded_model_src == ("tok/2", "repo/a")


# ============================================================================
# predict_all_batch forwards model_path/tokenizer_path
# ============================================================================

class TestPredictAllBatchForwardsModelPath:
    def test_forwards_model_path_to_prepare_model_switch(self, monkeypatch):
        captured = {}

        def fake_prepare_model_switch(model_path=None, tokenizer_path=None):
            captured["model_path"] = model_path
            captured["tokenizer_path"] = tokenizer_path
            return (tokenizer_path or "tok-default", model_path or "mdl-default")

        monkeypatch.setattr(kairos_strategies, "_prepare_model_switch", fake_prepare_model_switch)

        result = kairos_strategies.predict_all_batch(
            {}, model_path="repo/finetuned", tokenizer_path="repo/finetuned-tok")

        assert captured["model_path"] == "repo/finetuned"
        assert captured["tokenizer_path"] == "repo/finetuned-tok"
        assert result == {}


# ============================================================================
# _prepare_model_switch — cheap bookkeeping only, no heavy loading
# ============================================================================

class TestPrepareModelSwitch:
    def test_clears_caches_and_updates_loaded_src_on_switch(self):
        kairos_strategies._loaded_model_src = ("tok/a", "repo/a")
        kairos_strategies._prediction_cache[("BTC-USD", "t0")] = ["fake_pred"]
        kairos_strategies._dist_cache[("BTC-USD", "t0")] = "fake_dist"

        requested = kairos_strategies._prepare_model_switch(model_path="repo/b")

        assert requested == ("NeoQuasar/Kronos-Tokenizer-base", "repo/b")
        assert kairos_strategies._loaded_model_src == requested
        assert kairos_strategies._prediction_cache == {}
        assert kairos_strategies._dist_cache == {}

    def test_does_not_touch_weights_loaded_src_or_predictor(self):
        # _prepare_model_switch must be pure bookkeeping: it must not
        # materialize any weights, so _weights_loaded_src/bt_predictor are
        # left exactly as they were (no from_pretrained-equivalent call).
        kairos_strategies._weights_loaded_src = None
        kairos_strategies.bt_predictor = None

        kairos_strategies._prepare_model_switch(model_path="repo/b")

        assert kairos_strategies._weights_loaded_src is None
        assert kairos_strategies.bt_predictor is None

    def test_noop_when_requested_src_matches_loaded_src(self):
        kairos_strategies._loaded_model_src = ("NeoQuasar/Kronos-Tokenizer-base", "repo/a")
        kairos_strategies._prediction_cache[("BTC-USD", "t0")] = ["fake_pred"]
        kairos_strategies._dist_cache[("BTC-USD", "t0")] = "fake_dist"

        requested = kairos_strategies._prepare_model_switch(model_path="repo/a")

        assert requested == ("NeoQuasar/Kronos-Tokenizer-base", "repo/a")
        # No switch needed -> caches must survive untouched.
        assert kairos_strategies._prediction_cache == {("BTC-USD", "t0"): ["fake_pred"]}
        assert kairos_strategies._dist_cache == {("BTC-USD", "t0"): "fake_dist"}

    def test_never_imports_or_touches_model_loading(self, monkeypatch):
        # Sabotage the heavy-loading path: if _prepare_model_switch ever
        # called into it, this would raise.
        import model as model_module

        def _boom(*a, **kw):
            raise AssertionError("_prepare_model_switch must not touch model loading")

        monkeypatch.setattr(model_module, "Kronos", type("X", (), {"from_pretrained": staticmethod(_boom)}), raising=False)
        monkeypatch.setattr(model_module, "KronosTokenizer", type("Y", (), {"from_pretrained": staticmethod(_boom)}), raising=False)

        kairos_strategies._prepare_model_switch(model_path="repo/a")
        kairos_strategies._prepare_model_switch(model_path="repo/b")
        # No exception raised -> confirmed no heavy loading occurred.


# ============================================================================
# _materialize_model — no-op when weights already loaded for requested_src
# ============================================================================

class TestMaterializeModel:
    def test_noop_when_weights_already_loaded_for_requested_src(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        requested = ("NeoQuasar/Kronos-Tokenizer-base", "repo/a")
        kairos_strategies._materialize_model(requested)
        predictor_1 = kairos_strategies.bt_predictor

        def _boom(*a, **kw):
            raise AssertionError("from_pretrained should not be called again")

        import model as model_module
        monkeypatch.setattr(model_module.Kronos, "from_pretrained", classmethod(lambda cls, src: _boom()))

        kairos_strategies._materialize_model(requested)

        assert kairos_strategies.bt_predictor is predictor_1

    def test_loads_when_weights_loaded_src_differs(self, monkeypatch):
        _patch_model_loading(monkeypatch)
        kairos_strategies._materialize_model(("tok/1", "repo/a"))
        predictor_a = kairos_strategies.bt_predictor

        kairos_strategies._materialize_model(("tok/1", "repo/b"))

        assert kairos_strategies.bt_predictor is not predictor_a
        assert kairos_strategies._weights_loaded_src == ("tok/1", "repo/b")


# ============================================================================
# predict_all_batch — lazy materialization + fixed shared-cache-key bug
# ============================================================================

class TestPredictAllBatchLazyMaterialize:
    def _make_asset_df(self, n=310, base=100.0):
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "open": [base] * n, "high": [base + 1] * n, "low": [base - 1] * n,
            "close": [base + i * 0.01 for i in range(n)],
            "volume": [1000.0] * n,
        }, index=idx)

    def test_materialize_model_not_called_when_all_symbols_are_shared_cache_hits(self, monkeypatch, tmp_path):
        import kairos_predcache

        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        from kairos_backtest import KairosSettings
        monkeypatch.setattr(KairosSettings, "lookback", 300)
        monkeypatch.setattr(KairosSettings, "pred_samples", 5)
        monkeypatch.setattr(KairosSettings, "interval", "1d")

        df = self._make_asset_df()
        lookback_for_hash = min(KairosSettings.lookback, len(df))
        content_hash = kairos_predcache.content_hash_for_closes(
            df["close"].iloc[-lookback_for_hash:]
        )
        key = kairos_predcache.make_key(
            symbol="BTC-USD", interval="1d", bar_timestamp=df.index[-1],
            lookback_len=lookback_for_hash, pred_samples=5,
            model_id="NeoQuasar/Kronos-base", content_hash=content_hash, pred_len=1,
        )
        sample = pd.DataFrame({
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1.0], "amount": [1.0],
        }, index=[df.index[-1] + pd.Timedelta(days=1)])
        kairos_predcache.get_cache().put(key, [sample])

        def _boom(*a, **kw):
            raise AssertionError("should not be called")

        monkeypatch.setattr(kairos_strategies, "_materialize_model", _boom)

        result = kairos_strategies.predict_all_batch({"BTC-USD": df})

        assert "BTC-USD" in result
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

    def test_different_model_path_builds_different_shared_cache_key(self, monkeypatch, tmp_path):
        import kairos_predcache

        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        from kairos_backtest import KairosSettings
        monkeypatch.setattr(KairosSettings, "lookback", 300)
        monkeypatch.setattr(KairosSettings, "pred_samples", 5)
        monkeypatch.setattr(KairosSettings, "interval", "1d")
        monkeypatch.setattr(KairosSettings, "model", None)

        class _StubPredictor:
            def predict_batch(self, df_list, x_ts_list, y_ts_list, pred_len, sample_count,
                               return_samples, verbose):
                out = []
                for x_ts in x_ts_list:
                    samples = [pd.DataFrame({
                        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                        "volume": [1.0], "amount": [1.0],
                    }, index=[x_ts.iloc[-1]]) for _ in range(sample_count)]
                    out.append(samples)
                return out

        monkeypatch.setattr(kairos_strategies, "bt_predictor", _StubPredictor())
        monkeypatch.setattr(kairos_strategies, "_materialize_model", lambda *a, **kw: None)

        def fake_to_kronos_frame(df, lookback, amount="auto"):
            x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]].copy()
            x_ts = pd.Series(x_df.index)
            return x_df, x_ts

        def fake_future_timestamps(last_ts, interval, n, calendar, tz):
            return pd.Series([last_ts + pd.Timedelta(days=1)])

        monkeypatch.setattr(kairos_strategies, "to_kronos_frame", fake_to_kronos_frame)
        monkeypatch.setattr(kairos_strategies, "future_timestamps", fake_future_timestamps)

        df = self._make_asset_df()
        kairos_strategies.predict_all_batch({"BTC-USD": df}, model_path=None)
        files_after_base = set(os.listdir(tmp_path))

        kairos_strategies._prediction_cache.clear()
        kairos_strategies._dist_cache.clear()
        kairos_strategies.predict_all_batch({"BTC-USD": df}, model_path="repo/finetuned")
        files_after_finetuned = set(os.listdir(tmp_path))

        new_files = files_after_finetuned - files_after_base
        assert len(new_files) == 1, (
            "base and finetuned model_path must produce distinct shared-cache "
            "keys/files, not collide under KairosSettings.model"
        )

        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None


# ============================================================================
# is_batch_cached — read-only shared-cache precheck (no model load, no writes)
# ============================================================================

class TestIsBatchCached:
    def _make_asset_df(self, n=310, base=100.0):
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "open": [base] * n, "high": [base + 1] * n, "low": [base - 1] * n,
            "close": [base + i * 0.01 for i in range(n)],
            "volume": [1000.0] * n,
        }, index=idx)

    def test_false_when_shared_cache_inactive(self, monkeypatch):
        # KAIROS_PRED_CACHE_DIR unset -> kairos_predcache.get_cache() is None.
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        import kairos_predcache
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        df = self._make_asset_df()
        assert kairos_strategies.is_batch_cached({"BTC-USD": df}) is False

    def test_false_when_at_least_one_symbol_is_a_miss(self, monkeypatch, tmp_path):
        import kairos_predcache

        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        from kairos_backtest import KairosSettings
        monkeypatch.setattr(KairosSettings, "lookback", 300)
        monkeypatch.setattr(KairosSettings, "pred_samples", 5)
        monkeypatch.setattr(KairosSettings, "interval", "1d")

        df_a = self._make_asset_df()
        df_b = self._make_asset_df(base=50.0)

        # Only BTC-USD is prepopulated -- ETH-USD is a genuine miss.
        lookback_for_hash = min(KairosSettings.lookback, len(df_a))
        content_hash = kairos_predcache.content_hash_for_closes(
            df_a["close"].iloc[-lookback_for_hash:]
        )
        key = kairos_predcache.make_key(
            symbol="BTC-USD", interval="1d", bar_timestamp=df_a.index[-1],
            lookback_len=lookback_for_hash, pred_samples=5,
            model_id="NeoQuasar/Kronos-base", content_hash=content_hash, pred_len=1,
        )
        sample = pd.DataFrame({
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1.0], "amount": [1.0],
        }, index=[df_a.index[-1] + pd.Timedelta(days=1)])
        kairos_predcache.get_cache().put(key, [sample])

        result = kairos_strategies.is_batch_cached({"BTC-USD": df_a, "ETH-USD": df_b})
        assert result is False

        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

    def test_true_when_every_symbol_is_a_hit(self, monkeypatch, tmp_path):
        import kairos_predcache

        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        from kairos_backtest import KairosSettings
        monkeypatch.setattr(KairosSettings, "lookback", 300)
        monkeypatch.setattr(KairosSettings, "pred_samples", 5)
        monkeypatch.setattr(KairosSettings, "interval", "1d")

        df_a = self._make_asset_df()
        df_b = self._make_asset_df(base=50.0)

        sample = pd.DataFrame({
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1.0], "amount": [1.0],
        }, index=[df_a.index[-1] + pd.Timedelta(days=1)])

        for symbol, df in (("BTC-USD", df_a), ("ETH-USD", df_b)):
            lookback_for_hash = min(KairosSettings.lookback, len(df))
            content_hash = kairos_predcache.content_hash_for_closes(
                df["close"].iloc[-lookback_for_hash:]
            )
            key = kairos_predcache.make_key(
                symbol=symbol, interval="1d", bar_timestamp=df.index[-1],
                lookback_len=lookback_for_hash, pred_samples=5,
                model_id="NeoQuasar/Kronos-base", content_hash=content_hash, pred_len=1,
            )
            kairos_predcache.get_cache().put(key, [sample])

        result = kairos_strategies.is_batch_cached({"BTC-USD": df_a, "ETH-USD": df_b})
        assert result is True

        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

    def test_never_materializes_model_or_writes_to_cache(self, monkeypatch, tmp_path):
        """Read-only contract: no model load, no in-process _prediction_cache
        mutation, no shared-cache writes."""
        import kairos_predcache

        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        def _boom(*a, **kw):
            raise AssertionError("is_batch_cached must never materialize a model")

        monkeypatch.setattr(kairos_strategies, "_materialize_model", _boom)

        df = self._make_asset_df()
        result = kairos_strategies.is_batch_cached({"BTC-USD": df})

        assert result is False  # genuine miss, but no exception raised above
        assert kairos_strategies._prediction_cache == {}
        files_written = list(tmp_path.iterdir())
        assert files_written == []

        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None


# ============================================================================
# _shared_cache_key — refactor must not change predict_all_batch's actual
# cache keys (extracted from its former inline key-building logic)
# ============================================================================

class TestSharedCacheKeyRefactor:
    def _make_asset_df(self, n=310, base=100.0):
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "open": [base] * n, "high": [base + 1] * n, "low": [base - 1] * n,
            "close": [base + i * 0.01 for i in range(n)],
            "volume": [1000.0] * n,
        }, index=idx)

    def test_predict_all_batch_finds_hit_populated_via_shared_cache_key_helper(self, monkeypatch, tmp_path):
        """A shared-cache entry populated using _shared_cache_key's own
        output is found by predict_all_batch's real cache-lookup path --
        proving the extracted helper builds the exact same key string the
        inline logic used to build."""
        import kairos_predcache

        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None

        from kairos_backtest import KairosSettings
        monkeypatch.setattr(KairosSettings, "lookback", 300)
        monkeypatch.setattr(KairosSettings, "pred_samples", 5)
        monkeypatch.setattr(KairosSettings, "interval", "1d")

        df = self._make_asset_df()
        key = kairos_strategies._shared_cache_key(
            "BTC-USD", df, "NeoQuasar/Kronos-base", 1
        )
        sample = pd.DataFrame({
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1.0], "amount": [1.0],
        }, index=[df.index[-1] + pd.Timedelta(days=1)])
        kairos_predcache.get_cache().put(key, [sample])

        def _boom(*a, **kw):
            raise AssertionError("should not be called -- must be a cache hit")

        monkeypatch.setattr(kairos_strategies, "_materialize_model", _boom)

        result = kairos_strategies.predict_all_batch({"BTC-USD": df})

        assert "BTC-USD" in result

        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        kairos_predcache._singleton = None
        kairos_predcache._singleton_dir = None
