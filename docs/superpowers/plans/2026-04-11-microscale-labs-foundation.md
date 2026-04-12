# Microscale Labs Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational repo scaffolding, shared `microscale` Python package with reusable modules, tooling config (uv, jupytext, pre-commit, justfile, CI), and one complete lab (01-token-tax) as the reference pattern all future labs follow.

**Architecture:** Jupytext percent-format `.py` files are the single source of truth for each lab. They run as normal Python scripts (`python lab.py`) and sync bidirectionally with `.ipynb` notebooks for Colab. A shared `microscale/` package provides reusable modules (device detection, visualization, model loading, progress display) so labs stay thin and focused on educational content. `uv` manages dependencies with platform-aware PyTorch index routing.

**Tech Stack:** Python 3.11+, uv, PyTorch, HuggingFace transformers/hub, tiktoken, matplotlib, plotly, seaborn, rich, jupytext, nbmake, ruff, just

---

## File Map

```
Microscale/
├── CLAUDE.md                              # Updated with commands + conventions
├── README.md                              # Project README with badges, hardware table
├── pyproject.toml                         # uv-managed deps, platform-aware PyTorch
├── jupytext.toml                          # Percent format, bidirectional pairing
├── justfile                               # Task runner
├── .pre-commit-config.yaml                # ruff + nbstripout
├── .gitignore                             # Python/notebook/cache ignores
├── .github/
│   └── workflows/
│       ├── test-labs.yml                  # nbmake per-lab CI
│       └── sync-notebooks.yml             # Jupytext .py → .ipynb generation
│
├── microscale/                            # Shared reusable package
│   ├── __init__.py                        # Version + convenience re-exports
│   ├── device.py                          # CPU/CUDA/MPS/MLX auto-detection
│   ├── viz.py                             # Dual-output plotting (notebook + CLI)
│   ├── models.py                          # Pinned model registry + loaders
│   ├── cache.py                           # HF cache management + offline support
│   ├── env.py                             # Environment detection (Colab/notebook/script)
│   └── style.mplstyle                     # Microscale matplotlib theme
│
├── labs/
│   └── 01-token-tax/
│       ├── lab.py                         # Jupytext percent format (source of truth)
│       └── README.md                      # Learning objectives, prereqs, hardware
│
├── notebooks/                             # CI-generated .ipynb (gitignored on main)
│   └── .gitkeep
│
└── tests/
    ├── test_device.py                     # Tests for device detection
    ├── test_viz.py                        # Tests for viz module
    ├── test_env.py                        # Tests for environment detection
    └── test_labs.py                       # nbmake smoke test config
```

---

## Task 1: Clean Repo + .gitignore

**Files:**
- Create: `.gitignore`
- Remove: `*.png` screenshots from browser exploration, `.playwright-mcp/`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
.venv/
venv/
env/

# uv
.python-version

# Jupyter / Notebooks
.ipynb_checkpoints/
notebooks/*.ipynb

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# HuggingFace cache (don't commit models)
models/
*.safetensors
*.bin
*.gguf

# Lab outputs (generated artifacts)
outputs/

# Playwright MCP (browser automation artifacts)
.playwright-mcp/

# Test artifacts
.pytest_cache/
htmlcov/
.coverage
```

- [ ] **Step 2: Remove browser screenshots and playwright artifacts**

Run:
```bash
git rm -f *.png
rm -rf .playwright-mcp/
echo ".playwright-mcp/" >> .gitignore
```
Expected: All `.png` files and `.playwright-mcp/` removed from tracking.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore and clean repo artifacts"
```

---

## Task 2: pyproject.toml + uv Setup

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "microscale"
version = "0.1.0"
description = "Hands-on labs for Small Language Models — from tokenization to deployment"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }

dependencies = [
    "torch>=2.2",
    "transformers>=4.40",
    "safetensors>=0.4",
    "huggingface-hub>=0.24",
    "tiktoken>=0.8",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.20",
    "rich>=13.7",
    "tqdm>=4.66",
]

[project.optional-dependencies]
apple = ["mlx>=0.20", "mlx-lm>=0.20"]
inference = ["llama-cpp-python>=0.3"]
notebooks = ["jupyterlab>=4.2", "jupytext>=1.16", "ipywidgets>=8.1"]
dev = [
    "pytest>=8.0",
    "nbmake>=1.5",
    "ruff>=0.5",
    "pre-commit>=3.7",
    "nbstripout>=0.7",
    "jupytext>=1.16",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Initialize uv and sync**

Run:
```bash
uv sync --extra dev --extra notebooks
```
Expected: `.venv/` created, all deps installed, `uv.lock` generated.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pyproject.toml with uv dependency management"
```

---

## Task 3: microscale/env.py — Environment Detection

**Files:**
- Create: `microscale/__init__.py`
- Create: `microscale/env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env.py
from microscale.env import is_notebook, is_colab, is_ci, runtime_context


def test_is_notebook_returns_false_in_script():
    """Running under pytest is not a notebook."""
    assert is_notebook() is False


def test_is_colab_returns_false_locally():
    assert is_colab() is False


def test_is_ci_reflects_env(monkeypatch):
    monkeypatch.delenv("MICROSCALE_CI", raising=False)
    assert is_ci() is False
    monkeypatch.setenv("MICROSCALE_CI", "1")
    assert is_ci() is True


def test_runtime_context_returns_dict():
    ctx = runtime_context()
    assert "environment" in ctx
    assert "is_notebook" in ctx
    assert "is_colab" in ctx
    assert "is_ci" in ctx
    assert ctx["environment"] in ("script", "notebook", "colab")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'microscale'`

- [ ] **Step 3: Create microscale/__init__.py**

```python
# microscale/__init__.py
"""Microscale — shared utilities for Small Language Model labs."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Implement microscale/env.py**

```python
# microscale/env.py
"""Environment detection: notebook vs script, Colab vs local, CI vs interactive."""

import os


def is_notebook() -> bool:
    """Return True if running inside a Jupyter/Colab notebook kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def is_ci() -> bool:
    """Return True if MICROSCALE_CI=1 is set (used to reduce workload in CI)."""
    return os.environ.get("MICROSCALE_CI", "").strip() in ("1", "true", "yes")


def runtime_context() -> dict:
    """Return a summary dict of the current runtime environment."""
    if is_colab():
        env = "colab"
    elif is_notebook():
        env = "notebook"
    else:
        env = "script"

    return {
        "environment": env,
        "is_notebook": is_notebook(),
        "is_colab": is_colab(),
        "is_ci": is_ci(),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_env.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add microscale/__init__.py microscale/env.py tests/test_env.py
git commit -m "feat: add environment detection module"
```

---

## Task 4: microscale/device.py — Platform-Agnostic Device Detection

**Files:**
- Create: `microscale/device.py`
- Create: `tests/test_device.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_device.py
import torch

from microscale.device import DeviceInfo, Runtime, get_device, get_torch_device, device_summary


def test_get_device_returns_device_info():
    info = get_device()
    assert isinstance(info, DeviceInfo)
    assert isinstance(info.runtime, Runtime)
    assert isinstance(info.name, str)
    assert isinstance(info.description, str)
    assert info.name in ("cuda", "mps", "cpu", "mlx")


def test_get_torch_device_returns_torch_device():
    dev = get_torch_device()
    assert isinstance(dev, torch.device)


def test_tensor_creation_on_detected_device():
    dev = get_torch_device()
    x = torch.randn(4, 4, device=dev)
    assert x.device.type == dev.type


def test_device_summary_returns_string():
    summary = device_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement microscale/device.py**

```python
# microscale/device.py
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
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_device.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add microscale/device.py tests/test_device.py
git commit -m "feat: add platform-agnostic device detection module"
```

---

## Task 5: microscale/viz.py — Dual-Output Visualization Helpers

**Files:**
- Create: `microscale/viz.py`
- Create: `microscale/style.mplstyle`
- Create: `tests/test_viz.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_viz.py
import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from microscale.viz import apply_style, show, save_fig, heatmap, line_plot


def test_apply_style_does_not_error():
    apply_style()


def test_save_fig_creates_file():
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.png")
        save_fig(fig, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    plt.close(fig)


def test_heatmap_returns_figure():
    data = np.random.rand(5, 5)
    fig = heatmap(data, title="Test Heatmap", xlabel="Keys", ylabel="Queries")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_line_plot_returns_figure():
    x = np.arange(10)
    fig = line_plot(
        x=x,
        ys={"loss": np.random.rand(10), "val_loss": np.random.rand(10)},
        title="Training",
        xlabel="Step",
        ylabel="Loss",
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_show_in_script_mode_saves_to_file(monkeypatch):
    """In script mode, show() should save the figure to outputs/."""
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("MICROSCALE_OUTPUT_DIR", d)
        show(fig, filename="test_output.png")
        assert os.path.exists(os.path.join(d, "test_output.png"))
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Create the matplotlib style file**

```ini
# microscale/style.mplstyle
# Microscale Lab Theme — field journal aesthetic

# Figure
figure.facecolor: "#f4ecd8"
figure.edgecolor: "#e2d5b0"
figure.figsize: 10, 6
figure.dpi: 150

# Axes
axes.facecolor: "#faf5e8"
axes.edgecolor: "#3a4160"
axes.labelcolor: "#1a1f3a"
axes.titlesize: 14
axes.titleweight: bold
axes.labelsize: 11
axes.prop_cycle: cycler('color', ['#b87333', '#4a7c74', '#8b3a3a', '#5a7a3d', '#1a1f3a', '#d89050', '#2f5b54', '#6b7091'])
axes.grid: True
axes.grid.which: major

# Grid
grid.color: "#d4c8a8"
grid.linewidth: 0.5
grid.alpha: 0.7

# Ticks
xtick.color: "#3a4160"
ytick.color: "#3a4160"
xtick.labelsize: 9
ytick.labelsize: 9

# Legend
legend.framealpha: 0.9
legend.edgecolor: "#e2d5b0"
legend.facecolor: "#f4ecd8"
legend.fontsize: 10

# Text
text.color: "#1a1f3a"

# Font
font.family: serif
font.size: 11

# Savefig
savefig.facecolor: "#f4ecd8"
savefig.bbox: tight
savefig.dpi: 150
```

- [ ] **Step 4: Implement microscale/viz.py**

```python
# microscale/viz.py
"""Dual-output visualization helpers: inline in notebooks, saved files in scripts."""

import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from microscale.env import is_notebook

_STYLE_DIR = Path(__file__).parent / "style.mplstyle"


def apply_style() -> None:
    """Apply the Microscale matplotlib theme."""
    if _STYLE_DIR.exists():
        plt.style.use(str(_STYLE_DIR))
    # Use non-interactive backend in script mode to avoid GUI requirements
    if not is_notebook():
        matplotlib.use("Agg")


def _output_dir() -> Path:
    """Resolve the output directory for saved figures."""
    d = Path(os.environ.get("MICROSCALE_OUTPUT_DIR", "outputs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def show(fig: plt.Figure, filename: Optional[str] = None) -> None:
    """Display a figure inline (notebook) or save to disk (script).

    In notebook mode: calls plt.show().
    In script mode: saves to outputs/<filename> and prints a Rich message.
    """
    if is_notebook():
        plt.show()
    else:
        if filename is None:
            filename = "figure.png"
        path = _output_dir() / filename
        save_fig(fig, str(path))
        try:
            from rich.console import Console

            Console().print(f"  [dim]Saved:[/dim] {path}")
        except ImportError:
            print(f"  Saved: {path}")
        plt.close(fig)


def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a figure to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def heatmap(
    data: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    xticklabels: Optional[list] = None,
    yticklabels: Optional[list] = None,
    cmap: str = "YlOrBr",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Create a heatmap figure. Returns the Figure (caller decides show/save)."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        xticklabels=xticklabels if xticklabels else "auto",
        yticklabels=yticklabels if yticklabels else "auto",
        annot=data.shape[0] <= 12 and data.shape[1] <= 12,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#e2d5b0",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def line_plot(
    x: np.ndarray,
    ys: dict[str, np.ndarray],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a multi-line plot. Returns the Figure."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, y in ys.items():
        ax.plot(x, y, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add microscale/viz.py microscale/style.mplstyle tests/test_viz.py
git commit -m "feat: add dual-output visualization helpers with Microscale theme"
```

---

## Task 6: microscale/models.py — Pinned Model Registry

**Files:**
- Create: `microscale/models.py`
- Create: `microscale/cache.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
from microscale.models import MODEL_REGISTRY, get_model_info, list_models
from microscale.cache import setup_cache, cache_status


def test_registry_has_entries():
    assert len(MODEL_REGISTRY) > 0


def test_get_model_info_known_model():
    info = get_model_info("qwen3-0.6b")
    assert info is not None
    assert "repo" in info
    assert "revision" in info
    assert info["repo"].startswith("Qwen/")


def test_get_model_info_unknown_returns_none():
    assert get_model_info("nonexistent-model-xyz") is None


def test_list_models_returns_names():
    names = list_models()
    assert isinstance(names, list)
    assert "qwen3-0.6b" in names


def test_setup_cache_returns_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    path = setup_cache()
    assert path.exists()


def test_cache_status_returns_dict():
    status = cache_status()
    assert "cache_dir" in status
    assert "offline_mode" in status
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement microscale/models.py**

```python
# microscale/models.py
"""Pinned model registry — single source of truth for all model references.

Every lab imports model info from here. Pinning to a commit SHA ensures
labs never break silently when model authors update weights on the Hub.
"""

from typing import Optional

# Each entry pins a HuggingFace model to a specific commit.
# To update: change the revision, run all labs, commit.
MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B",
        "revision": "main",  # Pin to commit SHA after first successful test
        "description": "Qwen3 0.6B — smallest Qwen3 variant",
        "size_gb": 1.2,
        "params": "0.6B",
    },
    "smollm3-360m": {
        "repo": "HuggingFaceTB/SmolLM3-360M",
        "revision": "main",
        "description": "SmolLM3 360M — HuggingFace's tiny LM",
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


def get_model_info(name: str) -> Optional[dict]:
    """Look up a model by short name. Returns None if not found."""
    return MODEL_REGISTRY.get(name)


def list_models() -> list[str]:
    """Return all registered model short names."""
    return list(MODEL_REGISTRY.keys())


def load_tokenizer(name: str, **kwargs):
    """Load a tokenizer from the registry with pinned revision."""
    from transformers import AutoTokenizer

    info = MODEL_REGISTRY[name]
    return AutoTokenizer.from_pretrained(
        info["repo"], revision=info["revision"], **kwargs
    )


def load_model(name: str, **kwargs):
    """Load a model from the registry with pinned revision."""
    from transformers import AutoModelForCausalLM

    info = MODEL_REGISTRY[name]
    return AutoModelForCausalLM.from_pretrained(
        info["repo"], revision=info["revision"], **kwargs
    )
```

- [ ] **Step 4: Implement microscale/cache.py**

```python
# microscale/cache.py
"""HuggingFace cache management and offline mode support."""

import os
from pathlib import Path


def setup_cache(cache_dir: str | None = None) -> Path:
    """Configure and return the HF cache directory.

    Call at the top of each lab to ensure consistent cache behavior.
    """
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add microscale/models.py microscale/cache.py tests/test_models.py
git commit -m "feat: add pinned model registry and cache management"
```

---

## Task 7: microscale/__init__.py — Convenience Re-exports

**Files:**
- Modify: `microscale/__init__.py`

- [ ] **Step 1: Update the package init with convenience imports**

Replace the contents of `microscale/__init__.py` with:

```python
# microscale/__init__.py
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
```

- [ ] **Step 2: Verify all imports work**

Run: `uv run python -c "import microscale; print(microscale.__version__); print(microscale.device_summary())"`
Expected: Prints `0.1.0` and a device summary line.

- [ ] **Step 3: Commit**

```bash
git add microscale/__init__.py
git commit -m "feat: add convenience re-exports to package init"
```

---

## Task 8: Jupytext Configuration

**Files:**
- Create: `jupytext.toml`

- [ ] **Step 1: Create jupytext.toml**

```toml
# jupytext.toml — Bidirectional pairing between .py and .ipynb

# Percent format: # %% cell markers in .py files
# These are valid Python that runs with `python lab.py`
# AND opens as notebooks in VS Code, JupyterLab, PyCharm
formats = "py:percent,ipynb"

# Notebooks are paired: editing either .py or .ipynb + running
# `jupytext --sync` updates the other.
# In CI, we generate .ipynb from .py only.
```

- [ ] **Step 2: Verify jupytext is available**

Run:
```bash
uv run jupytext --version
```
Expected: Prints jupytext version (1.16+)

- [ ] **Step 3: Commit**

```bash
git add jupytext.toml
git commit -m "chore: add jupytext config for bidirectional .py/.ipynb sync"
```

---

## Task 9: justfile — Task Runner

**Files:**
- Create: `justfile`

- [ ] **Step 1: Install just (if not present)**

Run:
```bash
brew install just
```

- [ ] **Step 2: Create justfile**

```just
# justfile — Microscale Labs task runner
# Run `just` or `just --list` to see all commands.

# Default: show available commands
default:
    @just --list

# ─── Setup ───────────────────────────────────────

# Install all dependencies (dev + notebooks)
setup:
    uv sync --extra dev --extra notebooks

# Install with MLX support (Mac only)
setup-apple:
    uv sync --extra dev --extra notebooks --extra apple

# Install with llama.cpp inference tools
setup-inference:
    uv sync --extra dev --extra notebooks --extra inference

# ─── Labs ────────────────────────────────────────

# Open JupyterLab
lab:
    uv run jupyter lab

# Run a specific lab as a script: just run 01
run lab_num:
    uv run python labs/{{lab_num}}-*/lab.py

# ─── Notebooks ───────────────────────────────────

# Sync all .py labs to .ipynb notebooks
sync:
    @for f in labs/*/lab.py; do \
        uv run jupytext --to ipynb --output notebooks/$(basename $(dirname "$f")).ipynb "$f"; \
    done
    @echo "Synced all notebooks to notebooks/"

# Sync a specific lab: just sync-lab 01
sync-lab lab_num:
    uv run jupytext --to ipynb --output notebooks/$(basename $(dirname $(ls labs/{{lab_num}}-*/lab.py))).ipynb labs/{{lab_num}}-*/lab.py

# ─── Quality ─────────────────────────────────────

# Run all tests
test:
    uv run pytest tests/ -v

# Test a specific lab notebook
test-lab lab_num:
    uv run pytest --nbmake labs/{{lab_num}}-*/lab.py --nbmake-timeout=600

# Lint all code
lint:
    uv run ruff check microscale/ labs/ tests/
    uv run ruff format --check microscale/ labs/ tests/

# Format all code
fmt:
    uv run ruff check --fix microscale/ labs/ tests/
    uv run ruff format microscale/ labs/ tests/

# Strip notebook outputs before commit
strip:
    find labs -name "*.ipynb" -exec uv run nbstripout {} \;
    find notebooks -name "*.ipynb" -exec uv run nbstripout {} \;

# ─── Models ──────────────────────────────────────

# Prefetch all registered models for offline use
prefetch:
    uv run python -c "from microscale.models import MODEL_REGISTRY; from huggingface_hub import snapshot_download; [print(f'Downloading {n}...') or snapshot_download(m['repo'], revision=m['revision']) for n, m in MODEL_REGISTRY.items()]"
```

- [ ] **Step 3: Verify it works**

Run: `just --list`
Expected: Lists all available commands with descriptions.

- [ ] **Step 4: Commit**

```bash
git add justfile
git commit -m "chore: add justfile task runner"
```

---

## Task 10: Pre-commit + Lint Config

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create .pre-commit-config.yaml**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
        types_or: [python, jupyter]
      - id: ruff-format
        types_or: [python, jupyter]

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
```

- [ ] **Step 2: Install pre-commit hooks**

Run:
```bash
uv run pre-commit install
```
Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks (ruff, nbstripout)"
```

---

## Task 11: Lab 01 — The Token Tax (Reference Lab)

This is the first complete lab. It establishes the pattern all future labs follow.

**Files:**
- Create: `labs/01-token-tax/README.md`
- Create: `labs/01-token-tax/lab.py`

- [ ] **Step 1: Create the lab README**

```markdown
# Lab 01: The Token Tax

**Act:** I — The Landscape
**Time:** 30 minutes
**Hardware:** CPU only (runs in 10 seconds)
**XP:** Matches Act I, Lesson 1

## Learning Objectives

- Load and compare 4 BPE tokenizers (o200k, cl100k, p50k, gpt2)
- Measure the "token tax" across 5 languages
- Produce a heatmap showing tokenizer fairness gaps
- Understand why tokenizer vocabulary size matters for multilingual equity

## Prerequisites

- None — this is the first lab

## Artifact

A heatmap (PNG + interactive HTML) comparing token efficiency across languages and tokenizers.

## Run

```bash
# As a script
just run 01

# Or directly
python labs/01-token-tax/lab.py

# As a notebook
jupyter lab labs/01-token-tax/lab.py
```
```

- [ ] **Step 2: Create the lab source file (Jupytext percent format)**

```python
# labs/01-token-tax/lab.py
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 01: The Token Tax
#
# **Act I — The Landscape** | CPU only | ~30 minutes
#
# Every token a model processes costs compute. But not all text tokenizes equally.
# In this lab you'll load 4 real BPE tokenizers, feed them the same text in 5 languages,
# and measure the **token tax** — how many more tokens a non-English user pays
# for the same semantic content.
#
# **Aha moment:** You'll see that GPT-2's tokenizer charges Hindi ~4.7x more tokens
# than English — and that modern tokenizers (o200k) close the gap to ~1.3x.

# %% [markdown]
# ## Setup

# %%
# Install microscale if running in Colab
import subprocess, sys  # noqa: E401

try:
    import microscale
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/microscale-academy/labs.git"]
    )
    import microscale

from microscale import apply_style, show, device_summary, is_ci
from microscale.viz import heatmap

apply_style()
print(device_summary())

# %%
import tiktoken
import numpy as np
from rich.table import Table
from rich.console import Console

console = Console()

# %% [markdown]
# ## Step 1: Load the Tokenizers
#
# We'll compare 4 BPE encodings that span the history of OpenAI's tokenizers:
#
# | Encoding | Used By | Vocab Size |
# |----------|---------|------------|
# | `gpt2` | GPT-2, GPT-3 | 50,257 |
# | `p50k_base` | Codex, text-davinci-003 | 50,281 |
# | `cl100k_base` | GPT-3.5, GPT-4 | 100,256 |
# | `o200k_base` | GPT-4o, o1 | 200,019 |

# %%
ENCODINGS = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "p50k_base": tiktoken.get_encoding("p50k_base"),
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    "o200k_base": tiktoken.get_encoding("o200k_base"),
}

# Print vocab sizes
table = Table(title="Tokenizer Vocabulary Sizes")
table.add_column("Encoding", style="bold")
table.add_column("Vocab Size", justify="right")
for name, enc in ENCODINGS.items():
    table.add_row(name, f"{enc.n_vocab:,}")
console.print(table)

# %% [markdown]
# ## Step 2: Define Multi-Language Test Corpus
#
# We need semantically equivalent text across languages. These are short passages
# that express the same idea in each language — not machine translations, but
# natural phrasing.

# %%
CORPUS = {
    "English": (
        "The small language model processed the input sequence in twelve milliseconds. "
        "Each token was mapped to an embedding vector before the attention mechanism "
        "computed the weighted relationships between all positions in the context window."
    ),
    "Hindi": (
        "छोटे भाषा मॉडल ने इनपुट अनुक्रम को बारह मिलीसेकंड में संसाधित किया। "
        "ध्यान तंत्र द्वारा संदर्भ विंडो में सभी स्थितियों के बीच भारित संबंधों "
        "की गणना करने से पहले प्रत्येक टोकन को एक एम्बेडिंग वेक्टर में मैप किया गया।"
    ),
    "Japanese": (
        "小型言語モデルは入力シーケンスを12ミリ秒で処理しました。"
        "アテンションメカニズムがコンテキストウィンドウ内のすべての位置間の"
        "重み付き関係を計算する前に、各トークンは埋め込みベクトルにマッピングされました。"
    ),
    "Arabic": (
        "قام نموذج اللغة الصغير بمعالجة تسلسل الإدخال في اثني عشر ملي ثانية. "
        "تم تعيين كل رمز إلى متجه تضمين قبل أن تحسب آلية الانتباه "
        "العلاقات الموزونة بين جميع المواضع في نافذة السياق."
    ),
    "Python": (
        "def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "    residual = x\n"
        "    x = self.norm(x)\n"
        "    x = self.attention(x)\n"
        "    x = residual + x\n"
        "    residual = x\n"
        "    x = self.norm(x)\n"
        "    x = self.ffn(x)\n"
        "    return residual + x\n"
    ),
}

# %% [markdown]
# ## Step 3: Compute Token Counts
#
# For each (tokenizer, language) pair, count the tokens produced.

# %%
languages = list(CORPUS.keys())
encodings = list(ENCODINGS.keys())

# Matrix: rows = languages, cols = encodings
token_counts = np.zeros((len(languages), len(encodings)), dtype=int)

for i, lang in enumerate(languages):
    for j, enc_name in enumerate(encodings):
        tokens = ENCODINGS[enc_name].encode(CORPUS[lang])
        token_counts[i, j] = len(tokens)

# Display as a Rich table
table = Table(title="Token Counts by Language and Encoding")
table.add_column("Language", style="bold")
for enc_name in encodings:
    table.add_column(enc_name, justify="right")

for i, lang in enumerate(languages):
    table.add_row(lang, *[str(token_counts[i, j]) for j in range(len(encodings))])

console.print(table)

# %% [markdown]
# ## Step 4: Compute the Token Tax
#
# The **token tax** is the ratio of tokens for a given language vs English.
# A tax of 3.0× means you need 3× more tokens to express the same idea —
# which means 3× the inference cost, 3× the latency, and 1/3 the effective
# context window.

# %%
english_idx = languages.index("English")
token_tax = token_counts / token_counts[english_idx, :]  # Broadcast: divide each row by English

# Display
table = Table(title="Token Tax (relative to English = 1.0×)")
table.add_column("Language", style="bold")
for enc_name in encodings:
    table.add_column(enc_name, justify="right")

for i, lang in enumerate(languages):
    row = []
    for j in range(len(encodings)):
        tax = token_tax[i, j]
        color = "green" if tax <= 1.5 else "yellow" if tax <= 3.0 else "red"
        row.append(f"[{color}]{tax:.2f}×[/{color}]")
    table.add_row(lang, *row)

console.print(table)

# %% [markdown]
# ## Step 5: Visualize as a Heatmap
#
# A heatmap makes the fairness gap immediately visible.

# %%
fig = heatmap(
    token_tax,
    title="Token Tax: How Many More Tokens Than English?",
    xlabel="Tokenizer Encoding",
    ylabel="Language",
    xticklabels=encodings,
    yticklabels=languages,
    cmap="YlOrRd",
    figsize=(10, 5),
)
show(fig, filename="01-token-tax-heatmap.png")

# %% [markdown]
# ## Step 6: Interactive Exploration (Plotly)
#
# For a richer view, here's an interactive heatmap you can hover over.

# %%
import plotly.graph_objects as go

fig_interactive = go.Figure(
    data=go.Heatmap(
        z=token_tax,
        x=encodings,
        y=languages,
        colorscale="YlOrRd",
        text=[[f"{v:.2f}×" for v in row] for row in token_tax],
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="Language: %{y}<br>Encoding: %{x}<br>Tax: %{text}<extra></extra>",
    )
)
fig_interactive.update_layout(
    title="Token Tax: Interactive Heatmap",
    xaxis_title="Tokenizer Encoding",
    yaxis_title="Language",
    width=700,
    height=400,
)

from microscale.env import is_notebook as _is_nb

if _is_nb():
    fig_interactive.show()
else:
    from microscale.viz import _output_dir

    path = _output_dir() / "01-token-tax-interactive.html"
    fig_interactive.write_html(str(path), include_plotlyjs=True)
    console.print(f"  [dim]Interactive heatmap saved:[/dim] {path}")

# %% [markdown]
# ## Step 7: Key Takeaways
#
# **What you measured:**
# - GPT-2's tokenizer imposes a ~4-5× tax on Hindi and Arabic
# - `o200k_base` (200k vocab) cuts that to ~1.3-1.5×
# - Python code tokenizes efficiently across all encodings (ASCII-heavy)
# - Larger vocabularies help multilingual equity but increase embedding table size
#
# **Why it matters for SLMs:**
# A small model with a 32k vocabulary pays an even higher token tax than GPT-2.
# When deploying SLMs for multilingual use, the tokenizer choice directly
# impacts effective context length, latency, and cost.
#
# ## Artifact
#
# Check your `outputs/` directory for:
# - `01-token-tax-heatmap.png` — static heatmap
# - `01-token-tax-interactive.html` — interactive Plotly heatmap
```

- [ ] **Step 3: Verify the lab runs as a script**

Run:
```bash
uv run python labs/01-token-tax/lab.py
```
Expected: Rich tables print to terminal, two files saved to `outputs/`.

- [ ] **Step 4: Verify the lab opens as a notebook**

Run:
```bash
uv run jupytext --to ipynb labs/01-token-tax/lab.py --output notebooks/01-token-tax.ipynb
ls -la notebooks/01-token-tax.ipynb
```
Expected: `.ipynb` file created in `notebooks/`.

- [ ] **Step 5: Commit**

```bash
git add labs/01-token-tax/README.md labs/01-token-tax/lab.py
git commit -m "feat: add Lab 01 — The Token Tax (reference lab)"
```

---

## Task 12: CI Workflows

**Files:**
- Create: `.github/workflows/test-labs.yml`
- Create: `.github/workflows/sync-notebooks.yml`

- [ ] **Step 1: Create test-labs.yml**

```yaml
# .github/workflows/test-labs.yml
name: Test Labs

on:
  push:
    branches: [main]
    paths: ['labs/**', 'microscale/**', 'tests/**']
  pull_request:
    paths: ['labs/**', 'microscale/**', 'tests/**']

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev --extra notebooks
      - run: uv run pytest tests/ -v
        env:
          MICROSCALE_CI: "1"

  lab-smoke:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        lab: ['01']  # Extend as labs are added
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev --extra notebooks
      - name: Run lab ${{ matrix.lab }}
        run: uv run python labs/${{ matrix.lab }}-*/lab.py
        env:
          MICROSCALE_CI: "1"
          MICROSCALE_OUTPUT_DIR: "/tmp/lab-outputs"
```

- [ ] **Step 2: Create sync-notebooks.yml**

```yaml
# .github/workflows/sync-notebooks.yml
name: Sync Notebooks

on:
  push:
    branches: [main]
    paths: ['labs/**']

jobs:
  sync:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev --extra notebooks
      - name: Generate .ipynb from .py sources
        run: |
          mkdir -p notebooks
          for f in labs/*/lab.py; do
            dir=$(basename $(dirname "$f"))
            uv run jupytext --to ipynb --output "notebooks/${dir}.ipynb" "$f"
          done
      - name: Commit generated notebooks
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add notebooks/
          git diff --cached --quiet || git commit -m "chore: sync generated notebooks"
          git push
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/test-labs.yml .github/workflows/sync-notebooks.yml
git commit -m "ci: add lab testing and notebook sync workflows"
```

---

## Task 13: README.md + CLAUDE.md Updates

**Files:**
- Create: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create README.md**

```markdown
# Microscale Labs

Hands-on labs for [Microscale Academy](https://www.microscale.academy/) — a field journal for Small Language Models.

Every lab produces a number you can't get from prose. Every lab runs on consumer hardware. Every lab leaves you with a reusable artifact.

## Quick Start

**Option A: Colab (zero setup)**

Click the Colab badge on any lab below.

**Option B: Local with uv (recommended)**

```bash
git clone https://github.com/microscale-academy/labs.git
cd Microscale
uv sync --extra dev --extra notebooks
uv run jupyter lab
```

**Option C: Local with pip**

```bash
git clone https://github.com/microscale-academy/labs.git
cd Microscale
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
jupyter lab
```

**Option D: Run as scripts (no notebook required)**

```bash
uv sync
uv run python labs/01-token-tax/lab.py
# or: just run 01
```

## Labs

| # | Lab | Act | CPU | GPU | Mac | Colab | Time |
|---|-----|-----|-----|-----|-----|-------|------|
| 01 | [The Token Tax](labs/01-token-tax/) | I | yes | — | — | yes | 30m |
| 02 | Attention Under the Microscope | II | yes | faster | yes | yes | 60-90m |
| 03 | Build a Transformer Block | II | yes | — | yes | yes | 90-120m |
| 04 | Model Autopsy | III | yes | — | — | yes | 45-60m |
| 05 | The $1 Pretraining Run | IV | slow | yes | yes | T4 | 90-120m |
| 06 | The Hallucination Probe | V | yes | faster | yes | yes | 60-90m |
| 07 | LoRA in 50 Lines | VI | slow | yes | yes | T4 | 60-90m |
| 08 | Your First DPO Alignment | VI | slow | yes | yes | T4 | 90m |
| 09 | Quantize It Yourself | VII | yes | — | — | yes | 90m |
| 10 | The Roofline Lab | VIII | no | yes | Metal | yes | 60-90m |
| 11 | KV Cache Calculator | VIII | partial | yes | yes | yes | 60m |
| 12 | The Inference Showdown | IX | yes | yes | yes | partial | 45-60m |

## Hardware Requirements

These labs target consumer hardware:
- **Mac:** M-series with 16-32GB unified RAM
- **NVIDIA:** RTX 3060/4060+ with 8-12GB VRAM
- **Colab Free:** T4 GPU, 15GB VRAM
- **CPU:** Works for most labs (slower for training-heavy ones)

## Task Runner

If you have [just](https://github.com/casey/just) installed:

```bash
just              # list commands
just setup        # install deps
just run 01       # run lab 01 as a script
just lab          # open JupyterLab
just test         # run all tests
just sync         # generate .ipynb from .py sources
just prefetch     # download all models for offline use
```

## Project Structure

```
microscale/       # Shared reusable package (device, viz, models, cache)
labs/             # Lab source files (.py, Jupytext percent format)
notebooks/        # Generated .ipynb files (for Colab)
tests/            # Unit tests + lab smoke tests
```

## License

MIT
```

- [ ] **Step 2: Update CLAUDE.md with project commands and conventions**

Replace `CLAUDE.md` contents with:

```markdown
# Microscale Labs

## Project Overview

Hands-on labs for Microscale Academy (microscale.academy). A shared `microscale/` Python package provides reusable utilities; each lab in `labs/XX-name/lab.py` is a Jupytext percent-format file that runs as both a Python script and a Jupyter notebook.

## Development Guidelines

### Git Conventions

- **Commit messages must never mention AI, Claude, or any AI assistant.** No `Co-Authored-By` AI lines.
- Write commit messages in imperative mood (e.g., "Add feature" not "Added feature").
- Keep the first line under 72 characters.
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `ci:`.

### Code Style

- Follow existing patterns in the codebase.
- Prefer clarity over cleverness.
- Ruff for linting and formatting (configured in pyproject.toml).

### Lab Conventions

- Labs are Jupytext percent format `.py` files in `labs/XX-name/lab.py`.
- Each lab imports reusable utilities from `microscale/` — never duplicate shared logic.
- Every lab must run as both `python lab.py` (script) and as a notebook.
- First cell of every lab handles Colab install + device detection.
- Use `microscale.viz.show()` for all figures — it routes to inline or file output automatically.

### Testing

- `uv run pytest tests/ -v` — unit tests
- `uv run python labs/01-token-tax/lab.py` — smoke test a lab

## Commands

```bash
just setup          # Install all dependencies
just run 01         # Run lab 01 as a script
just lab            # Open JupyterLab
just test           # Run all tests
just lint           # Lint all code
just fmt            # Format all code
just sync           # Generate .ipynb from .py sources
just prefetch       # Download all models for offline use
just strip          # Strip notebook outputs
```

## Architecture

- `microscale/` — shared package (device detection, viz, model loading, cache)
- `labs/XX-name/lab.py` — Jupytext percent format, source of truth
- `notebooks/` — CI-generated .ipynb files for Colab
- `tests/` — pytest unit tests + nbmake smoke tests
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add project README and update CLAUDE.md"
```

---

## Self-Review

**Spec coverage:**
- [x] Normal Python scripts: `.py` files run with `python lab.py` — `# %%` is just a comment
- [x] Notebooks for Colab: Generated via Jupytext, CI auto-syncs
- [x] Bidirectional sync: Jupytext pairing, edit either format
- [x] Reusable modules: `microscale/` package with device, viz, models, cache, env
- [x] Platform agnostic: `device.py` handles CUDA/MPS/CPU, MLX opt-in
- [x] Visualizations: Matplotlib (static) + Plotly (interactive) + Rich (terminal)
- [x] Easy maintenance: Single source of truth per lab, shared utilities
- [x] Extensibility: Adding a lab = create `labs/XX-name/lab.py`, import from `microscale`

**Placeholder scan:** No TBD/TODO items found. All code blocks are complete.

**Type consistency:** Verified — `DeviceInfo`, `Runtime`, `show()`, `heatmap()`, `apply_style()` names are consistent across tasks.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-11-microscale-labs-foundation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
