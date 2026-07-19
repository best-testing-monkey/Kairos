"""Tests for kairos_strategies' model-switch-capable singleton loader.

No real model/GPU/network is touched: model.Kronos/KronosTokenizer/
KronosPredictor and kairos_gpu.ensure_cuda are monkeypatched with fakes.
_model_switch_needed is a pure function and is tested directly; the actual
load path (HF downloads, quantization) stays untested, matching the rest
of the codebase.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

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
    def test_forwards_model_path_to_ensure_model_loaded(self, monkeypatch):
        captured = {}

        def fake_ensure_model_loaded(model_path=None, tokenizer_path=None):
            captured["model_path"] = model_path
            captured["tokenizer_path"] = tokenizer_path
            # Leave bt_predictor as None; the test only checks the
            # assets-empty short-circuit path (no cached/uncached work),
            # so predict_all_batch never touches bt_predictor.

        monkeypatch.setattr(kairos_strategies, "_ensure_model_loaded", fake_ensure_model_loaded)

        result = kairos_strategies.predict_all_batch(
            {}, model_path="repo/finetuned", tokenizer_path="repo/finetuned-tok")

        assert captured["model_path"] == "repo/finetuned"
        assert captured["tokenizer_path"] == "repo/finetuned-tok"
        assert result == {}
