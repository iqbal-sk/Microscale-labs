"""Microscale — shared utilities for Small Language Model labs."""

__version__ = "0.1.0"

# Convenience re-exports so labs can do:
#   from microscale import get_device, show, apply_style
from microscale.device import get_device, get_torch_device, device_summary, DeviceInfo, Runtime
from microscale.env import is_notebook, is_colab, is_ci, runtime_context
from microscale.viz import apply_style, show, save_fig, heatmap, line_plot
from microscale.models import get_model_info, list_models, load_tokenizer, load_model
from microscale.cache import setup_cache, cache_status

__all__ = [
    # device
    "get_device", "get_torch_device", "device_summary", "DeviceInfo", "Runtime",
    # env
    "is_notebook", "is_colab", "is_ci", "runtime_context",
    # viz
    "apply_style", "show", "save_fig", "heatmap", "line_plot",
    # models
    "get_model_info", "list_models", "load_tokenizer", "load_model",
    # cache
    "setup_cache", "cache_status",
]
