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
    """Apply the Microscale matplotlib theme.

    Note: does NOT change the matplotlib backend. We use IPython's
    display() directly in show() to avoid backend-related rendering
    issues on Colab and other environments.
    """
    # Only set Agg backend in script mode AND if pyplot hasn't been
    # imported yet (which it has been by line 8 of this file). This is
    # mostly a no-op now, kept for backwards compatibility.
    if not is_notebook():
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass

    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))


def _output_dir() -> Path:
    """Resolve the output directory for saved figures."""
    d = Path(os.environ.get("MICROSCALE_OUTPUT_DIR", "outputs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def show(fig: plt.Figure, filename: str | None = None) -> None:
    """Display a figure inline (notebook) or save to disk (script).

    In notebook mode: uses IPython's display() directly — works in
    Jupyter, Colab, and VS Code regardless of matplotlib backend.
    In script mode: saves to outputs/<filename> and prints a Rich message.
    """
    if is_notebook():
        # Use IPython display() directly. This bypasses the matplotlib
        # backend entirely and works on Colab where backend detection
        # via plt.show() can be flaky.
        try:
            from IPython.display import display

            display(fig)
            plt.close(fig)  # prevent Jupyter from also showing it
            return
        except ImportError:
            # Fall back to plt.show() if IPython unavailable
            plt.show()
            return

    # Script mode: save to disk
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
