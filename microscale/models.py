# microscale/models.py
"""Pinned model registry — single source of truth for all model references.

Every lab imports model info from here. Pinning to a commit SHA ensures
labs never break silently when model authors update weights on the Hub.
"""

MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B",
        "revision": "main",
        "description": "Qwen3 0.6B — smallest Qwen3 variant",
        "size_gb": 1.2,
        "params": "0.6B",
    },
    "smollm2-360m": {
        "repo": "HuggingFaceTB/SmolLM2-360M",
        "revision": "main",
        "description": "SmolLM2 360M — HuggingFace's tiny LM",
        "size_gb": 0.7,
        "params": "360M",
    },
    "smollm3-3b": {
        "repo": "HuggingFaceTB/SmolLM3-3B",
        "revision": "main",
        "description": "SmolLM3 3B — mid-range small model",
        "size_gb": 6.0,
        "params": "3B",
    },
    "phi-4-mini": {
        "repo": "microsoft/phi-4-mini",
        "revision": "main",
        "description": "Phi-4 Mini 3.8B — Microsoft's compact model",
        "size_gb": 7.6,
        "params": "3.8B",
    },
}


def get_model_info(name: str) -> dict | None:
    """Look up a model by short name. Returns None if not found."""
    return MODEL_REGISTRY.get(name)


def list_models() -> list[str]:
    """Return all registered model short names."""
    return list(MODEL_REGISTRY.keys())


def load_tokenizer(name: str, **kwargs):
    """Load a tokenizer from the registry with pinned revision."""
    from transformers import AutoTokenizer

    info = MODEL_REGISTRY[name]
    return AutoTokenizer.from_pretrained(info["repo"], revision=info["revision"], **kwargs)


def load_model(name: str, **kwargs):
    """Load a model from the registry with pinned revision."""
    from transformers import AutoModelForCausalLM

    info = MODEL_REGISTRY[name]
    return AutoModelForCausalLM.from_pretrained(info["repo"], revision=info["revision"], **kwargs)
