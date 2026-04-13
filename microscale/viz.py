# microscale/viz.py
"""Dual-output visualization helpers: inline in notebooks, saved files in scripts."""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from microscale.env import is_notebook

_STYLE_PATH = Path(__file__).parent / "style.mplstyle"


def apply_style() -> None:
    """Apply the Microscale matplotlib theme."""
    if is_notebook():
        # Notebook: ensure inline backend so plots show in the output cell.
        # On Colab, matplotlib sometimes defaults to 'agg' without the
        # %matplotlib inline magic. Set it explicitly here.
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("matplotlib", "inline")
        except (ImportError, AttributeError):
            pass
    else:
        # Script: use Agg (non-interactive) to avoid needing a display.
        matplotlib.use("Agg")

    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))


def _output_dir() -> Path:
    """Resolve the output directory for saved figures."""
    d = Path(os.environ.get("MICROSCALE_OUTPUT_DIR", "outputs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def show(fig: plt.Figure, filename: str | None = None) -> None:
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
    xticklabels: list | None = None,
    yticklabels: list | None = None,
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
