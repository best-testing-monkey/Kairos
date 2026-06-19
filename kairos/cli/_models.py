"""Model registry: maps model short-names to HuggingFace IDs and config."""

MODELS: dict[str, dict] = {
    "kronos-mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "kronos-small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "kronos-base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}


def resolve(name: str) -> dict:
    """Return the model config for *name*, or raise ValueError."""
    if name in MODELS:
        return MODELS[name]
    # Allow a HF repo ID to be passed directly (owner/repo form)
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


def load_predictor(model_name: str, device: str = "cpu"):
    """Load and return a KronosPredictor for *model_name* on *device*."""
    import sys
    import os
    # Ensure the repo root is on the path so `model` is importable
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

    cfg = resolve(model_name)
    print(f"Loading tokenizer  {cfg['tokenizer_id']} …")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    print(f"Loading model      {cfg['model_id']} …")
    kronos = Kronos.from_pretrained(cfg["model_id"])
    predictor = KronosPredictor(
        kronos, tokenizer, device=device, max_context=cfg["max_context"]
    )
    return predictor
