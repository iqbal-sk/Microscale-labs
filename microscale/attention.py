# microscale/attention.py
"""Attention extraction, visualization, classification, and ablation utilities.

Designed for educational use: extract attention weights from HuggingFace models,
visualize patterns, classify head types, and measure ablation impact.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_attention(
    model,
    tokenizer,
    text: str,
    device: torch.device | None = None,
) -> dict:
    """Run a forward pass and return attention weights for all layers.

    IMPORTANT: The model must be loaded with `attn_implementation='eager'`.
    SDPA and FlashAttention do not support `output_attentions=True`.

    Args:
        model: A HuggingFace CausalLM loaded with attn_implementation='eager'.
        tokenizer: The matching tokenizer.
        text: Input text string.
        device: Torch device. If None, uses model's device.

    Returns:
        dict with:
            "weights": list of numpy arrays, shape (num_heads, seq_len, seq_len)
            "tokens": list of token strings
            "input_ids": the input_ids tensor
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len)
    weights = [attn[0].cpu().float().numpy() for attn in outputs.attentions]
    tokens = [tokenizer.decode(t) for t in inputs.input_ids[0].tolist()]

    return {
        "weights": weights,  # list of (num_heads, seq_len, seq_len)
        "tokens": tokens,
        "input_ids": inputs.input_ids,
    }


# ---------------------------------------------------------------------------
# Head pattern metrics
# ---------------------------------------------------------------------------


def head_entropy(attn_matrix: np.ndarray) -> float:
    """Compute average entropy across rows of an attention matrix.

    Higher entropy = more uniform attention. Lower = more concentrated.

    Args:
        attn_matrix: (seq_len, seq_len) post-softmax attention weights.
    """
    # Clip to avoid log(0)
    p = np.clip(attn_matrix, 1e-10, 1.0)
    row_entropies = -np.sum(p * np.log2(p), axis=-1)
    return float(np.mean(row_entropies))


def sink_strength(attn_matrix: np.ndarray) -> float:
    """Measure how much attention flows to the first token (position 0).

    Returns the mean attention weight on column 0 across all query positions.
    High value (> 0.3) indicates a "sink" head.
    """
    return float(np.mean(attn_matrix[:, 0]))


def prev_token_strength(attn_matrix: np.ndarray) -> float:
    """Measure how much attention flows to the immediately previous token.

    Returns the mean of the off-diagonal (position i attends to i-1).
    High value (> 0.3) indicates a "previous-token" head.
    """
    seq_len = attn_matrix.shape[0]
    if seq_len < 2:
        return 0.0
    # Gather attention[i, i-1] for i=1..seq_len-1
    diag_values = np.array([attn_matrix[i, i - 1] for i in range(1, seq_len)])
    return float(np.mean(diag_values))


def diagonal_strength(attn_matrix: np.ndarray) -> float:
    """Measure how much attention flows to the current token (self-attention).

    Returns the mean of the main diagonal.
    """
    return float(np.mean(np.diag(attn_matrix)))


def classify_head(attn_matrix: np.ndarray) -> str:
    """Classify an attention head's dominant pattern.

    Returns one of: "sink", "previous_token", "self", "uniform", "mixed"
    """
    s_sink = sink_strength(attn_matrix)
    s_prev = prev_token_strength(attn_matrix)
    s_diag = diagonal_strength(attn_matrix)
    entropy = head_entropy(attn_matrix)

    # Thresholds tuned for typical transformer behavior
    if s_sink > 0.3:
        return "sink"
    if s_prev > 0.3:
        return "previous_token"
    if s_diag > 0.3:
        return "self"
    if entropy > np.log2(attn_matrix.shape[0]) * 0.8:
        return "uniform"
    return "mixed"


def compute_head_summary(weights: list[np.ndarray]) -> dict:
    """Compute summary metrics for every head across all layers.

    Args:
        weights: list of (num_heads, seq_len, seq_len) arrays, one per layer.

    Returns:
        dict with numpy arrays of shape (num_layers, num_heads):
            "entropy", "sink", "prev_token", "diagonal", "classification"
    """
    n_layers = len(weights)
    n_heads = weights[0].shape[0]

    entropy = np.zeros((n_layers, n_heads))
    sink = np.zeros((n_layers, n_heads))
    prev_tok = np.zeros((n_layers, n_heads))
    diag = np.zeros((n_layers, n_heads))
    classifications = np.empty((n_layers, n_heads), dtype=object)

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            mat = weights[layer_idx][head_idx]
            entropy[layer_idx, head_idx] = head_entropy(mat)
            sink[layer_idx, head_idx] = sink_strength(mat)
            prev_tok[layer_idx, head_idx] = prev_token_strength(mat)
            diag[layer_idx, head_idx] = diagonal_strength(mat)
            classifications[layer_idx, head_idx] = classify_head(mat)

    return {
        "entropy": entropy,
        "sink": sink,
        "prev_token": prev_tok,
        "diagonal": diag,
        "classification": classifications,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_attention_head(
    attn_matrix: np.ndarray,
    tokens: list[str],
    title: str = "",
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    show_values: bool = False,
) -> plt.Axes:
    """Plot a single attention head as a heatmap.

    Args:
        attn_matrix: (seq_len, seq_len) attention weights.
        tokens: List of token strings for axis labels.
        title: Plot title.
        ax: Matplotlib axes. If None, creates a new figure.
        cmap: Colormap name. Default "viridis" (perceptually uniform, colorblind-safe).
        show_values: Whether to annotate cells with numeric values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attn_matrix, cmap=cmap, vmin=0, aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(title, fontsize=11)

    if show_values and len(tokens) <= 15:
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                val = attn_matrix[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_attention_overview(
    summary: dict,
    metric: str = "entropy",
    title: str = "",
    cmap: str | None = None,
) -> plt.Figure:
    """Plot a 2D heatmap (layers x heads) summarizing one metric per head.

    Args:
        summary: Output of compute_head_summary().
        metric: One of "entropy", "sink", "prev_token", "diagonal".
        title: Plot title.
        cmap: Colormap. Defaults vary by metric.
    """
    data = summary[metric]
    n_layers, n_heads = data.shape

    if cmap is None:
        cmap_map = {
            "entropy": "viridis",
            "sink": "YlOrRd",
            "prev_token": "YlGnBu",
            "diagonal": "PuBuGn",
        }
        cmap = cmap_map.get(metric, "viridis")

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.6), max(6, n_layers * 0.25)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    ax.set_title(title or f"Attention Head {metric.replace('_', ' ').title()}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_head_grid(
    weights: list[np.ndarray],
    tokens: list[str],
    layers: list[int] | None = None,
    cmap: str = "viridis",
    figsize_per_head: float = 1.2,
) -> plt.Figure:
    """Plot a grid of attention heatmaps for selected layers.

    Args:
        weights: list of (num_heads, seq_len, seq_len), one per layer.
        tokens: Token strings for axis labels.
        layers: Which layers to show. None = 5 evenly spaced layers.
        cmap: Colormap.
        figsize_per_head: Size multiplier per subplot.
    """
    n_all_layers = len(weights)
    n_heads = weights[0].shape[0]

    if layers is None:
        # Pick 5 evenly spaced layers
        layers = [int(i) for i in np.linspace(0, n_all_layers - 1, 5)]

    n_rows = len(layers)
    fig, axes = plt.subplots(
        n_rows,
        n_heads,
        figsize=(n_heads * figsize_per_head, n_rows * figsize_per_head),
        squeeze=False,
    )

    for row, layer_idx in enumerate(layers):
        for col in range(n_heads):
            ax = axes[row, col]
            ax.imshow(weights[layer_idx][col], cmap=cmap, vmin=0, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=8, rotation=0, labelpad=20)
            if row == 0:
                ax.set_title(f"H{col}", fontsize=7)

    fig.suptitle("Attention Patterns Across Layers and Heads", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


@contextmanager
def ablate_head(model, layer_idx: int, head_idx: int, head_dim: int = 128):
    """Context manager that zeros a specific attention head's output.

    Hooks into the attention layer's output projection input and zeros the
    slice corresponding to the target head before projection.

    Usage:
        with ablate_head(model, layer=5, head=3):
            outputs = model(**inputs)  # head 3 in layer 5 is zeroed

    Args:
        model: HuggingFace model with model.model.layers[i].self_attn.o_proj.
        layer_idx: Which layer (0-indexed).
        head_idx: Which head to ablate (0-indexed).
        head_dim: Dimension per head (128 for Qwen3-0.6B).
    """
    attn_module = model.model.layers[layer_idx].self_attn

    def hook_fn(module, args):
        # o_proj input shape: (batch, seq_len, num_heads * head_dim)
        x = args[0]
        start = head_idx * head_dim
        end = start + head_dim
        x[:, :, start:end] = 0.0
        return (x,) + args[1:]

    handle = attn_module.o_proj.register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
