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
