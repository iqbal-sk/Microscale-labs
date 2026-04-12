"""Platform-agnostic device detection for CPU, CUDA, MPS, and MLX."""

import sys
from dataclasses import dataclass
from enum import Enum


class Runtime(Enum):
    PYTORCH_CUDA = "pytorch-cuda"
    PYTORCH_MPS = "pytorch-mps"
    PYTORCH_CPU = "pytorch-cpu"
    MLX = "mlx"


@dataclass(frozen=True)
class DeviceInfo:
    runtime: Runtime
    name: str  # "cuda", "mps", "cpu", "mlx"
    description: str  # Human-readable summary


def get_device(prefer_mlx: bool = False) -> DeviceInfo:
    """Auto-detect the best available compute device.

    Priority: CUDA > MPS > CPU (default).
    If prefer_mlx=True on macOS: MLX > MPS > CPU.
    """
    if prefer_mlx and sys.platform == "darwin":
        try:
            import mlx.core as mx
            return DeviceInfo(
                runtime=Runtime.MLX,
                name="mlx",
                description=f"MLX on Apple Silicon ({mx.default_device()})",
            )
        except ImportError:
            pass

    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return DeviceInfo(
            runtime=Runtime.PYTORCH_CUDA,
            name="cuda",
            description=f"CUDA — {gpu_name} ({vram_gb:.1f} GB)",
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            runtime=Runtime.PYTORCH_MPS,
            name="mps",
            description="MPS (Metal Performance Shaders) on Apple Silicon",
        )

    return DeviceInfo(
        runtime=Runtime.PYTORCH_CPU,
        name="cpu",
        description="CPU (no GPU acceleration detected)",
    )


def get_torch_device() -> "torch.device":
    """Convenience: return a torch.device for the best available backend."""
    import torch
    info = get_device(prefer_mlx=False)
    return torch.device(info.name)


def device_summary() -> str:
    """Return a one-line summary of the detected device for lab headers."""
    info = get_device()
    return f"[{info.runtime.value}] {info.description}"
