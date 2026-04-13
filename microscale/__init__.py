"""Microscale — shared utilities for Small Language Model labs."""

__version__ = "0.1.0"

# Convenience re-exports so labs can do:
#   from microscale import get_device, show, apply_style
from microscale.attention import (
    ablate_head,
    classify_head,
    compute_head_summary,
    extract_attention,
    plot_attention_head,
    plot_attention_overview,
    plot_head_grid,
)
from microscale.cache import cache_status, setup_cache
from microscale.device import DeviceInfo, Runtime, device_summary, get_device, get_torch_device
from microscale.env import get_secret, is_ci, is_colab, is_notebook, runtime_context
from microscale.metrics import compute_per_token_loss, compute_perplexity
from microscale.models import get_model_info, list_models, load_model, load_tokenizer
from microscale.viz import apply_style, heatmap, line_plot, save_fig, show

__all__ = [
    # device
    "get_device",
    "get_torch_device",
    "device_summary",
    "DeviceInfo",
    "Runtime",
    # env
    "is_notebook",
    "is_colab",
    "is_ci",
    "get_secret",
    "runtime_context",
    # viz
    "apply_style",
    "show",
    "save_fig",
    "heatmap",
    "line_plot",
    # models
    "get_model_info",
    "list_models",
    "load_tokenizer",
    "load_model",
    # cache
    "setup_cache",
    "cache_status",
    # metrics
    "compute_perplexity",
    "compute_per_token_loss",
    # attention
    "extract_attention",
    "compute_head_summary",
    "classify_head",
    "plot_attention_head",
    "plot_attention_overview",
    "plot_head_grid",
    "ablate_head",
]
