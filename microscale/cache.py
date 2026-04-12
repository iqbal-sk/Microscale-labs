# microscale/cache.py
"""HuggingFace cache management and offline mode support."""

import os
from pathlib import Path


def setup_cache(cache_dir: str | None = None) -> Path:
    """Configure and return the HF cache directory."""
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def enable_offline() -> None:
    """Enable offline mode — only use cached models, no downloads."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def cache_status() -> dict:
    """Return current cache configuration."""
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return {
        "cache_dir": str(cache_dir),
        "exists": cache_dir.exists(),
        "offline_mode": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
