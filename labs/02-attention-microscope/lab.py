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
# # Lab 02: Attention Under the Microscope
#
# **Act II — Inside the Machine** | CPU or Apple Silicon | ~60–90 minutes
#
# ---
#
# ### What you will learn
#
# By the end of this lab you will be able to:
#
# 1. **Extract** raw attention weight matrices from a real 596M-parameter language model
# 2. **Visualize** what 448 attention heads actually look like — and read the patterns
# 3. **Identify** three documented head types: attention sinks, previous-token heads,
#    and broad-context heads
# 4. **Observe** how Grouped Query Attention (GQA) makes paired heads share structure
# 5. **Measure** which heads the model depends on by ablating them one at a time
# 6. **Conclude**, from your own data, that most heads are redundant — the empirical
#    basis for structured pruning
#
# ---
#
# ### The idea
#
# When a transformer processes text, every token gets to "look at" every earlier
# token through attention. But what does that attention actually look like? Are all
# heads doing the same thing? Do they all matter equally?
#
# You are about to find out — not from a diagram, but from the actual weight
# matrices of a production language model running on your machine.

# %% [markdown]
# ---
# ## 1. Setup

# %%
import subprocess
import sys

try:
    import microscale
except ImportError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/microscale-academy/labs.git",
        ]
    )
    import microscale

from microscale import apply_style, device_summary, get_torch_device, is_ci, show
from microscale.attention import (
    ablate_head,
    compute_head_summary,
    extract_attention,
    plot_attention_head,
    plot_head_grid,
)
from microscale.metrics import compute_perplexity

apply_style()
device = get_torch_device()
print(device_summary())

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

# %% [markdown]
# ---
# ## 2. Meet the Model
#
# We are working with **Qwen3-0.6B** — a 596M-parameter language model. Here is
# what matters for this lab:
#
# | | |
# |---|---|
# | **Layers** | 28 transformer blocks stacked in sequence |
# | **Query heads** | 16 per layer — each asks "what should I attend to?" |
# | **Key-Value heads** | 8 per layer — shared between pairs of query heads (GQA) |
# | **Head dimension** | 128 values per head |
# | **Total heads** | 28 layers × 16 heads = **448 attention matrices** |
#
# The **Grouped Query Attention** design means heads 0–1 share key/value
# projections, heads 2–3 share another pair, and so on. This saves memory
# while letting each query head specialize through its own projection.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3-0.6B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float32,
    attn_implementation="eager",  # needed to return attention weights
).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Ready: {n_params:.0f}M parameters on {device}")

# %% [markdown]
# ---
# ## 3. Extract the Attention Weights
#
# We feed one sentence through the model and capture every attention matrix.
# Each matrix tells us: for each token, how much did it attend to every
# other token?

# %%
INPUT_TEXT = (
    "The scientist observed the experiment carefully and recorded the results in her notebook"
)

result = extract_attention(model, tokenizer, INPUT_TEXT, device=device)
weights = result["weights"]  # 28 arrays, each (16, seq_len, seq_len)
tokens = result["tokens"]

n_layers = len(weights)
n_heads = weights[0].shape[0]
seq_len = weights[0].shape[1]

console.print(
    f"\n  Extracted [bold]{n_layers * n_heads}[/bold] attention matrices"
    f"  ({n_layers} layers × {n_heads} heads)"
    f"  for [bold]{seq_len}[/bold] tokens\n"
)
console.print(f"  Tokens: {' · '.join(tokens)}")

# %% [markdown]
# ---
# ## 4. The Landscape — All 448 Heads at a Glance
#
# Before zooming in, let's see the big picture. For every head we compute three
# numbers:
#
# - **Sink strength** — how much attention goes to the very first token?
# - **Previous-token strength** — how much goes to the token right before?
# - **Entropy** — how spread out is the attention distribution?
#
# Each metric becomes a 28×16 pixel in the heatmaps below. One pixel = one head.

# %%
summary = compute_head_summary(weights)

fig, axes = plt.subplots(1, 3, figsize=(20, 9))

panels = [
    ("sink", "How much attention goes\nto the first token?", "Sink Strength", "YlOrRd"),
    (
        "prev_token",
        "How much attention goes\nto the previous token?",
        "Previous-Token Strength",
        "YlGnBu",
    ),
    ("entropy", "How spread out\nis the attention?", "Entropy (bits)", "cividis"),
]

for ax, (metric, subtitle, label, cmap) in zip(axes, panels):
    data = summary[metric]
    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Head Index", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(0, n_heads, 2))
    ax.set_yticks(range(0, n_layers, 4))
    ax.set_title(f"{label}\n{subtitle}", fontsize=10, linespacing=1.4)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

fig.suptitle(
    "448 Attention Heads at a Glance",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()
show(fig, filename="02-attention-landscape.png")

# %% [markdown]
# **What to notice:**
#
# - The **sink panel** (left) is warm almost everywhere — the majority of heads
#   send significant attention to position 0. This is the "attention sink"
#   phenomenon: the first token acts as a safe no-op target when a head has
#   nothing semantically useful to attend to.
#
# - The **previous-token panel** (center) has a few bright spots, mostly in
#   early layers. These are specialist heads that pass "what came before me"
#   information forward — a building block of the induction circuit.
#
# - The **entropy panel** (right) varies widely. Low-entropy heads are
#   specialists; high-entropy heads attend broadly across the context.

# %% [markdown]
# ---
# ## 5. Zooming In — Three Pattern Types
#
# Let's find one head of each type and look at the full attention matrix
# with token labels.

# %%
sink_scores = summary["sink"]
prev_scores = summary["prev_token"]
entropy_scores = summary["entropy"]

sink_layer, sink_head = np.unravel_index(np.argmax(sink_scores), sink_scores.shape)
prev_layer, prev_head = np.unravel_index(np.argmax(prev_scores), prev_scores.shape)
entropy_layer, entropy_head = np.unravel_index(np.argmax(entropy_scores), entropy_scores.shape)

# %%
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

examples = [
    (
        sink_layer,
        sink_head,
        f"Sink Head  (Layer {sink_layer}, Head {sink_head})",
        "Notice the bright column on the left —\nevery token attends to the first position.",
        "magma",
    ),
    (
        prev_layer,
        prev_head,
        f"Previous-Token Head  (Layer {prev_layer}, Head {prev_head})",
        "Bright diagonal one step below center —\neach token looks at the one before it.",
        "magma",
    ),
    (
        entropy_layer,
        entropy_head,
        f"Broad-Context Head  (Layer {entropy_layer}, Head {entropy_head})",
        "Attention is spread broadly — no single\nposition dominates.",
        "magma",
    ),
]

for ax, (layer, head, title, annotation, cmap) in zip(axes, examples):
    plot_attention_head(
        weights[layer][head],
        tokens,
        title=title,
        ax=ax,
        cmap=cmap,
        show_values=seq_len <= 15,
    )
    # Add annotation below
    ax.text(
        0.5,
        -0.22,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        style="italic",
        color="#3a4160",
    )

fig.suptitle(
    "Three Types of Attention Heads",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.98])
show(fig, filename="02-attention-head-types.png")

# %% [markdown]
# **How to read these heatmaps:**
# - **Rows** = query tokens (the one doing the looking)
# - **Columns** = key tokens (the one being looked at)
# - **Brighter** = more attention weight (higher probability in the softmax)
#
# The causal mask means a token can only attend to earlier positions, which
# is why the upper-right triangle is dark.

# %% [markdown]
# ---
# ## 6. GQA in Action — Paired Heads
#
# In Grouped Query Attention, every two query heads share the same key and
# value projections. Let's place paired heads side by side to see if that
# sharing shows up visually.

# %%
pair_layer = prev_layer  # use a layer with interesting patterns
fig, axes = plt.subplots(2, 4, figsize=(22, 10))

for col in range(4):
    head_a = col * 2
    head_b = col * 2 + 1

    plot_attention_head(
        weights[pair_layer][head_a],
        tokens,
        title=f"Head {head_a}",
        ax=axes[0, col],
        cmap="magma",
    )
    plot_attention_head(
        weights[pair_layer][head_b],
        tokens,
        title=f"Head {head_b}",
        ax=axes[1, col],
        cmap="magma",
    )
    # Group label
    axes[0, col].text(
        0.5,
        1.18,
        f"KV Group {col}",
        transform=axes[0, col].transAxes,
        fontsize=9,
        ha="center",
        fontweight="bold",
        color="#4a7c74",
    )

fig.suptitle(
    f"GQA Pairs in Layer {pair_layer} — Same Keys & Values, Different Queries",
    fontsize=14,
    fontweight="bold",
    y=1.04,
)
fig.tight_layout()
show(fig, filename="02-gqa-pairs.png")

# %% [markdown]
# Each column is one KV group. The top and bottom heads share keys and values,
# but their queries differ — so the patterns are **similar but not identical**.
# That is the GQA trade-off: fewer KV heads saves memory, but each query head
# still gets its own perspective.

# %% [markdown]
# ---
# ## 7. Layer-by-Layer Evolution
#
# How do attention patterns change as we go deeper into the model?

# %%
fig = plot_head_grid(
    weights,
    tokens,
    layers=[0, 6, 13, 20, 27],
    cmap="magma",
    figsize_per_head=1.3,
)
fig.suptitle(
    "How Attention Evolves from Layer 0 to Layer 27",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
show(fig, filename="02-attention-grid.png")

# %% [markdown]
# - **Early layers** (top): simpler patterns — positional, previous-token, sinks
# - **Middle layers**: increasingly content-dependent
# - **Late layers** (bottom): complex, harder to interpret — these capture
#   semantic relationships that earlier layers built up

# %% [markdown]
# ---
# ## 8. Head Census — Classifying All 448 Heads
#
# Let's automatically classify every head and map them.

# %%
classifications = summary["classification"]

from collections import Counter

type_counts = Counter(classifications.flatten())

table = Table(title="Attention Head Census")
table.add_column("Pattern", style="bold")
table.add_column("Count", justify="right")
table.add_column("Fraction", justify="right")
table.add_column("Description")

descriptions = {
    "sink": "Attends primarily to the first token",
    "previous_token": "Attends to the immediately preceding token",
    "self": "Attends primarily to the current token",
    "uniform": "Spreads attention broadly across all positions",
    "mixed": "No single dominant pattern",
}

for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    frac = count / (n_layers * n_heads)
    table.add_row(
        ptype,
        str(count),
        f"{frac:.1%}",
        descriptions.get(ptype, ""),
    )
console.print(table)

# %%
# Visual classification map
fig, ax = plt.subplots(figsize=(14, 8))

type_to_num = {
    "sink": 0,
    "previous_token": 1,
    "self": 2,
    "uniform": 3,
    "mixed": 4,
}
colors = ["#b87333", "#4a7c74", "#5a7a3d", "#6b7091", "#d4c8a8"]
class_numeric = np.vectorize(lambda x: type_to_num.get(x, 4))(classifications)

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cmap_class = ListedColormap(colors)
ax.imshow(class_numeric, cmap=cmap_class, aspect="auto", interpolation="nearest")
ax.set_xlabel("Head Index", fontsize=11)
ax.set_ylabel("Layer", fontsize=11)
ax.set_title(
    "Classification Map — Every Head, Every Layer",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(range(n_heads))
ax.set_yticks(range(0, n_layers, 2))

legend_items = [
    Patch(facecolor=c, edgecolor="#3a4160", linewidth=0.5, label=name.replace("_", " ").title())
    for name, c in zip(type_to_num.keys(), colors)
]
ax.legend(
    handles=legend_items,
    loc="upper right",
    framealpha=0.95,
    fontsize=9,
    title="Pattern Type",
)
fig.tight_layout()
show(fig, filename="02-head-classification-map.png")

# %% [markdown]
# ---
# ## 9. Ablation — Which Heads Actually Matter?
#
# This is the experiment that connects everything. We will:
# 1. Measure the model's baseline perplexity (how well it predicts text)
# 2. Zero out each head, one at a time, and re-measure
# 3. Record how much perplexity changes
#
# A head that raises perplexity sharply when removed is **load-bearing**.
# A head that barely moves the needle is **redundant**.

# %%
EVAL_TEXTS = [
    "The research team published their findings in a prestigious journal last month.",
    "Small language models are increasingly deployed on mobile devices and edge hardware.",
    "Attention allows each token to gather relevant context from earlier positions.",
    "Training on curated high quality data often matters more than sheer dataset size.",
    "The capital of France is Paris and the capital of Japan is Tokyo.",
]

print("Measuring baseline perplexity...")
baseline_ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS, device=device)
console.print(f"\n  Baseline perplexity: [bold]{baseline_ppl:.2f}[/bold]")

# %%
N_LAYERS_TO_ABLATE = n_layers if not is_ci() else 4
layers_to_ablate = list(range(N_LAYERS_TO_ABLATE))
HEAD_DIM = 128

console.print(
    f"\n  Running {N_LAYERS_TO_ABLATE * n_heads} ablation experiments"
    f" ({N_LAYERS_TO_ABLATE} layers × {n_heads} heads)..."
)

ablation_delta = np.zeros((N_LAYERS_TO_ABLATE, n_heads))

from tqdm import tqdm

for i, layer_idx in enumerate(tqdm(layers_to_ablate, desc="Ablating")):
    for head_idx in range(n_heads):
        with ablate_head(model, layer_idx, head_idx, head_dim=HEAD_DIM):
            ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS, device=device)
        ablation_delta[i, head_idx] = ppl - baseline_ppl

# %%
# --- Ablation heatmap ---
fig, ax = plt.subplots(figsize=(16, max(7, N_LAYERS_TO_ABLATE * 0.3)))

# Use a log-ish scale to show both small and large effects
vmax = max(2.0, float(np.percentile(np.abs(ablation_delta), 97)))
im = ax.imshow(
    ablation_delta,
    cmap="RdYlGn_r",
    aspect="auto",
    vmin=-vmax * 0.1,
    vmax=vmax,
    interpolation="nearest",
)
ax.set_xlabel("Head Index", fontsize=11)
ax.set_ylabel("Layer", fontsize=11)
ax.set_title(
    "Ablation Impact — Perplexity Change When Each Head Is Zeroed",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xticks(range(n_heads))
ax.set_yticks(range(N_LAYERS_TO_ABLATE))
ax.set_yticklabels([str(idx) for idx in layers_to_ablate])
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Perplexity change  (red = important, green = negligible)", fontsize=9)

# Mark the top-3 most important heads with stars
flat = [
    (layers_to_ablate[i], j, ablation_delta[i, j])
    for i in range(N_LAYERS_TO_ABLATE)
    for j in range(n_heads)
]
flat.sort(key=lambda x: -x[2])
for rank, (layer, head, delta) in enumerate(flat[:3]):
    row_idx = layers_to_ablate.index(layer)
    ax.plot(
        head,
        row_idx,
        marker="*",
        color="white",
        markersize=14,
        markeredgecolor="#1a1f3a",
        markeredgewidth=1.0,
    )

fig.tight_layout()
show(fig, filename="02-ablation-impact.png")

# %% [markdown]
# The white stars mark the three most important heads. Notice how they cluster
# in early layers — these are likely part of the **induction circuit**, the
# fundamental mechanism for in-context learning.

# %%
# --- Results table ---
top_k = 5

table = Table(title="Most Important Heads (largest perplexity increase)")
table.add_column("Rank", justify="center", style="bold")
table.add_column("Layer", justify="right")
table.add_column("Head", justify="right")
table.add_column("PPL Change", justify="right")
table.add_column("Verdict")

for rank, (layer, head, delta) in enumerate(flat[:top_k], 1):
    severity = "[red bold]critical[/]" if delta > 10 else "[yellow]notable[/]"
    table.add_row(
        str(rank),
        str(layer),
        str(head),
        f"+{delta:.1f}",
        severity,
    )
console.print(table)

# %%
# --- Summary statistic ---
threshold = 0.1
negligible = sum(1 for _, _, d in flat if abs(d) < threshold)
total = N_LAYERS_TO_ABLATE * n_heads
pct = negligible / total

console.print(
    f"\n  [bold]{negligible}[/bold] of {total} heads"
    f" ([bold]{pct:.0%}[/bold]) cause less than"
    f" {threshold} perplexity change when removed."
)
console.print("  This is why head pruning works: most heads are redundant.\n")

# %% [markdown]
# ---
# ## 10. Interactive Attention Explorer
#
# For a richer view, here is an interactive heatmap you can hover over.
# (In script mode this saves as an HTML file you can open in a browser.)

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Show the 3 most important heads interactively
fig_interactive = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        f"Layer {flat[i][0]}, Head {flat[i][1]} (+{flat[i][2]:.1f} PPL)" for i in range(3)
    ],
    horizontal_spacing=0.08,
)

for col_idx in range(3):
    layer, head, delta = flat[col_idx]
    mat = weights[layer][head]
    fig_interactive.add_trace(
        go.Heatmap(
            z=mat,
            x=tokens,
            y=tokens,
            colorscale="Magma",
            zmin=0,
            text=[[f"{v:.3f}" for v in row] for row in mat],
            hovertemplate=("Query: %{y}<br>Key: %{x}<br>Weight: %{text}<extra></extra>"),
            showscale=col_idx == 2,
        ),
        row=1,
        col=col_idx + 1,
    )
    fig_interactive.update_xaxes(
        tickangle=45,
        tickfont_size=9,
        row=1,
        col=col_idx + 1,
    )
    fig_interactive.update_yaxes(
        tickfont_size=9,
        row=1,
        col=col_idx + 1,
    )

fig_interactive.update_layout(
    title_text="Interactive: Three Most Important Attention Heads",
    title_font_size=15,
    width=1100,
    height=450,
    template="plotly_white",
)

from microscale.env import is_notebook as _is_nb

if _is_nb():
    fig_interactive.show()
else:
    from microscale.viz import _output_dir

    path = _output_dir() / "02-attention-interactive.html"
    fig_interactive.write_html(str(path), include_plotlyjs=True)
    console.print(f"  [dim]Interactive explorer saved:[/dim] {path}")

# %% [markdown]
# ---
# ## What You Learned
#
# | Finding | Evidence |
# |---|---|
# | Attention sinks are real | Most heads attend to position 0 |
# | Previous-token heads are early | A few early heads specialize in "look back" |
# | GQA pairs share structure | Paired heads are similar, not identical |
# | Most heads are redundant | Over half cause < 0.1 PPL change |
# | A few heads are critical | 2–3 early heads cause massive PPL increase |
#
# ### Artifacts in `outputs/`
#
# | File | What it shows |
# |------|---------------|
# | `02-attention-landscape.png` | 448-head overview: sink, previous-token, entropy |
# | `02-attention-head-types.png` | Annotated examples of three pattern types |
# | `02-gqa-pairs.png` | GQA paired heads side by side |
# | `02-attention-grid.png` | Layer-by-layer pattern evolution |
# | `02-head-classification-map.png` | Color-coded map of every head's type |
# | `02-ablation-impact.png` | Which heads matter (with star markers) |
# | `02-attention-interactive.html` | Hoverable Plotly explorer for top heads |
#
# ### References
#
# - Olsson et al., "In-context Learning and Induction Heads" (2022)
# - Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
# - Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
# - Zheng et al., "Attention Heads of Large Language Models: A Survey" (2025)
