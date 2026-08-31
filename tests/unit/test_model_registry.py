"""Tests for kairos/models.py -- the shared model registry moved out of
kairos/cli/_models.py so strategy/kairos_strategies.py can use it too."""
import pytest

from kairos.models import MODELS, resolve


class TestResolveShortNames:
    def test_mini(self):
        cfg = resolve("mini")
        assert cfg == {
            "model_id": "NeoQuasar/Kronos-mini",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
            "max_context": 2048,
        }

    def test_small(self):
        cfg = resolve("small")
        assert cfg == {
            "model_id": "NeoQuasar/Kronos-small",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        }

    def test_base(self):
        cfg = resolve("base")
        assert cfg == {
            "model_id": "NeoQuasar/Kronos-base",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        }


class TestKronosPrefixStrip:
    def test_kronos_base_still_resolves(self):
        # forecast.py/finetune.py usage strings document "--model kronos-base".
        assert resolve("kronos-base") == resolve("base")

    def test_kronos_small_still_resolves(self):
        assert resolve("kronos-small") == resolve("small")

    def test_kronos_mini_still_resolves(self):
        assert resolve("kronos-mini") == resolve("mini")


class TestReverseLookupByModelId:
    def test_mini_model_id_returns_max_context_2048(self):
        # Load-bearing: _materialize_model() only ever has the resolved
        # model_id (mdl_src), not the short name. Without this reverse
        # lookup, "NeoQuasar/Kronos-mini" would fall into the generic "/"
        # passthrough and silently get max_context=512 instead of 2048.
        cfg = resolve("NeoQuasar/Kronos-mini")
        assert cfg["max_context"] == 2048
        assert cfg["tokenizer_id"] == "NeoQuasar/Kronos-Tokenizer-2k"

    def test_small_model_id(self):
        assert resolve("NeoQuasar/Kronos-small") == resolve("small")

    def test_base_model_id(self):
        assert resolve("NeoQuasar/Kronos-base") == resolve("base")


class TestHfIdPassthrough:
    def test_unknown_hf_repo_id_passes_through(self):
        cfg = resolve("someorg/some-other-model")
        assert cfg == {
            "model_id": "someorg/some-other-model",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        }


class TestLocalCheckpointPassthrough:
    def test_local_absolute_path_passes_through(self):
        path = "/data/finetuned/BTC-USD_ETH-USD_1d"
        cfg = resolve(path)
        assert cfg == {
            "model_id": path,
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        }


class TestUnknownNameRaises:
    def test_bare_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve("not-a-real-model")

    def test_error_message_lists_short_names(self):
        with pytest.raises(ValueError) as exc_info:
            resolve("nope")
        msg = str(exc_info.value)
        for name in MODELS:
            assert name in msg
