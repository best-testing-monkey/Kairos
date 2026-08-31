"""Model registry re-export -- moved to kairos/models.py so it can be shared
by strategy/kairos_strategies.py without importing anything under cli/.
"""
from kairos.models import MODELS, resolve  # noqa: F401


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
