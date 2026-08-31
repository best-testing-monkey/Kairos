"""Model registry: maps model short-names to HuggingFace IDs and config.

Shared by kairos/cli/_models.py (forecast/finetune CLIs) and
strategy/kairos_strategies.py (backtest/papertrade prediction paths) --
one registry, not three independent hardcoded pairings.
"""

MODELS: dict[str, dict] = {
    "mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

_BY_MODEL_ID = {cfg["model_id"]: cfg for cfg in MODELS.values()}


def resolve(name: str) -> dict:
    """Return the model config for *name*, or raise ValueError.

    Resolution order:
      1. exact short-name match in MODELS ("mini"/"small"/"base")
      2. a "kronos-" prefix stripped and retried (back-compat: forecast.py/
         finetune.py usage strings document "--model kronos-base")
      3. reverse lookup by model_id -- callers that already hold a resolved
         HF id (e.g. kairos_strategies._materialize_model, which only ever
         sees mdl_src, never the original short name) must still get the
         right max_context/tokenizer back. Load-bearing for mini: without
         this, "NeoQuasar/Kronos-mini" would fall through to the generic
         HF-id passthrough below and silently get max_context=512 instead
         of 2048.
      4. if "/" in name: passthrough as an HF repo id / local checkpoint
         path, paired with the base tokenizer and max_context 512 -- what
         local finetuned checkpoint dirs and FINETUNE_BASE_MODEL rely on.
      5. otherwise raise ValueError.
    """
    if name in MODELS:
        return MODELS[name]

    if name.startswith("kronos-"):
        stripped = name[len("kronos-"):]
        if stripped in MODELS:
            return MODELS[stripped]

    if name in _BY_MODEL_ID:
        return _BY_MODEL_ID[name]

    if "/" in name:
        return {
            "model_id": name,
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        }

    raise ValueError(
        f"Unknown model {name!r}. Known names: {', '.join(MODELS)}. "
        "You can also pass a HuggingFace repo ID directly (e.g. owner/repo)."
    )
