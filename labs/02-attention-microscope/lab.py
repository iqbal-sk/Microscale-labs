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
# Every transformer prediction passes through attention layers where tokens
# decide how much to "look at" each other. In this lab you will crack open
# a real language model, extract the raw attention weights from all 448 heads,
# and see for yourself what those patterns look like.
#
# You will also ablate individual heads — zero them out — and measure whether
# the model even notices. The result is often surprising: most heads barely
# matter, and a few are load-bearing.
#
# **What you will produce:**
# - A gallery of attention heatmaps with annotated pattern types
# - An ablation impact map showing which heads the model depends on

# %% [markdown]
# ## Setup

# %%
import subprocess
import sys

try:
    import microscale
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/user/Microscale.git"]
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
# ## Part 1: Loading the Model
#
# We are using **Qwen3-0.6B**, a 596-million-parameter language model from Alibaba.
# A few things to know about its architecture:
#
# | Property | Value |
# |----------|-------|
# | Parameters | 596M |
# | Layers | 28 |
# | Query heads | 16 |
# | Key-Value heads | 8 (Grouped Query Attention) |
# | Head dimension | 128 |
# | Vocabulary | 151,936 tokens |
#
# The **Grouped Query Attention (GQA)** means every pair of query heads shares
# one key-value head. Heads 0 and 1 share KV, heads 2 and 3 share KV, and so on.
# You will see this in the attention patterns: paired heads often look similar
# because they attend to the same keys, just with different queries.
#
# We load with `attn_implementation="eager"` because the faster SDPA kernel
# does not support returning attention weights.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3-0.6B (this may take a minute on first run)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float32,  # float32 for stable attention extraction
    attn_implementation="eager",  # required for output_attentions=True
).to(device)
model.eval()
print(f"Loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters on {device}")

# %% [markdown]
# ## Part 2: Extracting Attention Weights
#
# Let's feed a simple sentence through the model and capture every attention
# matrix. With 28 layers and 16 heads per layer, that is **448 attention matrices**,
# each of shape (sequence_length × sequence_length).

# %%
INPUT_TEXT = (
    "The scientist observed the experiment carefully and recorded the results in her notebook"
)

result = extract_attention(model, tokenizer, INPUT_TEXT, device=device)
weights = result["weights"]  # list of 28 arrays, each (16, seq_len, seq_len)
tokens = result["tokens"]

n_layers = len(weights)
n_heads = weights[0].shape[0]
seq_len = weights[0].shape[1]

table = Table(title="Extraction Summary")
table.add_column("Property", style="bold")
table.add_column("Value", justify="right")
table.add_row("Input text", INPUT_TEXT[:60] + "...")
table.add_row("Tokens", str(seq_len))
table.add_row("Layers", str(n_layers))
table.add_row("Heads per layer", str(n_heads))
table.add_row("Total attention matrices", str(n_layers * n_heads))
table.add_row("Token list", " | ".join(tokens))
console.print(table)

# %% [markdown]
# ## Part 3: The Landscape View
#
# Before zooming into individual heads, let's get a bird's-eye view of all 448
# heads. We will compute summary metrics for each head:
#
# - **Entropy**: How spread out is the attention? High entropy means the head
#   attends broadly. Low entropy means it concentrates on a few positions.
# - **Sink strength**: How much attention flows to the first token (position 0)?
#   Many heads in language models use the first token as an "attention sink" —
#   a safe default when they have nothing useful to attend to.
# - **Previous-token strength**: How much attention flows to the immediately
#   preceding token? Some heads specialize in "look at what came just before me."

# %%
summary = compute_head_summary(weights)

# Show the three key metrics as 28x16 heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

metrics = [
    ("sink", "Sink Strength\n(attention to first token)", "YlOrRd"),
    ("prev_token", "Previous-Token Strength\n(attention to position i-1)", "YlGnBu"),
    ("entropy", "Entropy\n(how spread out is attention)", "viridis"),
]

for ax, (metric, title, cmap) in zip(axes, metrics):
    data = summary[metric]
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, n_heads, 2))
    ax.set_yticks(range(0, n_layers, 4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("The Attention Landscape: 448 Heads at a Glance", fontsize=13, y=1.02)
fig.tight_layout()
show(fig, filename="02-attention-landscape.png")

# %% [markdown]
# What do you notice?
#
# - **Sink strength** (left): Many heads across all layers show moderate-to-high
#   sink behavior (warm colors). This is the "attention sink" phenomenon, documented
#   in research from Xiao et al. (2023). The first token acts as a safe dumping
#   ground for attention when a head has nothing semantically useful to attend to.
#
# - **Previous-token strength** (middle): A few heads in early layers light up
#   strongly. These are the "previous-token heads" — they specialize in "what
#   token came right before me?", feeding positional information forward.
#
# - **Entropy** (right): Varies widely. Low-entropy heads are specialists
#   (concentrated attention). High-entropy heads attend broadly.

# %% [markdown]
# ## Part 4: Zooming In — Pattern Types
#
# Let's look at individual heads to see these patterns up close. We will pick
# one head of each type based on the metrics we just computed.

# %%
# Find the strongest sink head, previous-token head, and a high-entropy head
sink_scores = summary["sink"]
prev_scores = summary["prev_token"]
entropy_scores = summary["entropy"]

# Best of each type
sink_layer, sink_head = np.unravel_index(np.argmax(sink_scores), sink_scores.shape)
prev_layer, prev_head = np.unravel_index(np.argmax(prev_scores), prev_scores.shape)
entropy_layer, entropy_head = np.unravel_index(np.argmax(entropy_scores), entropy_scores.shape)

console.print(
    f"\n[bold]Strongest sink head:[/bold] Layer {sink_layer}, Head {sink_head}"
    f" (score: {sink_scores[sink_layer, sink_head]:.3f})"
)
console.print(
    f"[bold]Strongest previous-token head:[/bold] Layer {prev_layer}, Head {prev_head}"
    f" (score: {prev_scores[prev_layer, prev_head]:.3f})"
)
console.print(
    f"[bold]Highest entropy head:[/bold] Layer {entropy_layer}, Head {entropy_head}"
    f" (entropy: {entropy_scores[entropy_layer, entropy_head]:.3f})"
)

# %%
# Plot these three heads side by side
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

heads_to_show = [
    (sink_layer, sink_head, "Sink Head"),
    (prev_layer, prev_head, "Previous-Token Head"),
    (entropy_layer, entropy_head, "High-Entropy Head"),
]

for ax, (layer, head, label) in zip(axes, heads_to_show):
    plot_attention_head(
        weights[layer][head],
        tokens,
        title=f"{label}\nLayer {layer}, Head {head}",
        ax=ax,
        cmap="viridis",
        show_values=len(tokens) <= 15,
    )

fig.suptitle("Three Types of Attention Heads", fontsize=13, y=1.02)
fig.tight_layout()
show(fig, filename="02-attention-head-types.png")

# %% [markdown]
# **Reading these heatmaps:**
# - Each row is a **query** position (the token doing the looking)
# - Each column is a **key** position (the token being looked at)
# - Brighter = more attention weight
#
# In the **sink head**, you see a bright vertical column on the left — every
# token sends attention to position 0 regardless of content.
#
# In the **previous-token head**, you see a bright diagonal stripe one step
# below the main diagonal — each token attends most strongly to the token
# right before it.
#
# In the **high-entropy head**, attention is spread more broadly — no single
# position dominates.

# %% [markdown]
# ## Part 5: GQA Pairs — Shared Keys, Different Queries
#
# Because Qwen3-0.6B uses Grouped Query Attention, heads 0 and 1 share the
# same key and value projections, just with different query projections.
# Let's see if paired heads actually look similar.

# %%
# Pick a layer with interesting patterns (use the sink layer)
pair_layer = sink_layer
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col in range(4):
    head_a = col * 2  # Even head
    head_b = col * 2 + 1  # Odd head (shares KV with head_a)

    plot_attention_head(
        weights[pair_layer][head_a],
        tokens,
        title=f"Head {head_a} (KV group {col})",
        ax=axes[0, col],
        cmap="viridis",
    )
    plot_attention_head(
        weights[pair_layer][head_b],
        tokens,
        title=f"Head {head_b} (KV group {col})",
        ax=axes[1, col],
        cmap="viridis",
    )

fig.suptitle(
    f"GQA Pairs in Layer {pair_layer}: Same Keys, Different Queries",
    fontsize=13,
    y=1.02,
)
fig.tight_layout()
show(fig, filename="02-gqa-pairs.png")

# %% [markdown]
# Notice how paired heads (top and bottom in each column) share structural
# similarities — they attend to similar positions because they use the same
# keys and values — but the exact weights differ because each head has its
# own query projection.
#
# This is the GQA trade-off: you save memory and compute by sharing KV heads
# (8 instead of 16), but each query head can still specialize through its
# unique query projection.

# %% [markdown]
# ## Part 6: The Grid View
#
# Let's see a sample of heads across layers to watch how attention patterns
# evolve from early to late layers.

# %%
# Show 5 representative layers
fig = plot_head_grid(weights, tokens, layers=[0, 6, 13, 20, 27])
show(fig, filename="02-attention-grid.png")

# %% [markdown]
# Early layers (top rows) tend to show simpler patterns — positional attention,
# previous-token attention, or sinks. Deeper layers (bottom rows) show more
# content-dependent patterns that are harder to interpret but capture semantic
# relationships.

# %% [markdown]
# ## Part 7: Pattern Census
#
# Let's classify every head and see how the 448 heads break down by type.

# %%
classifications = summary["classification"]

# Count each type
from collections import Counter

type_counts = Counter(classifications.flatten())

table = Table(title="Attention Head Census (448 heads)")
table.add_column("Pattern Type", style="bold")
table.add_column("Count", justify="right")
table.add_column("Fraction", justify="right")

for pattern_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    table.add_row(pattern_type, str(count), f"{count / (n_layers * n_heads):.1%}")

console.print(table)

# Show classification map
fig, ax = plt.subplots(figsize=(12, 8))
type_to_num = {"sink": 0, "previous_token": 1, "self": 2, "uniform": 3, "mixed": 4}
class_numeric = np.vectorize(lambda x: type_to_num.get(x, 4))(classifications)

from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#b87333", "#4a7c74", "#5a7a3d", "#6b7091", "#d4c8a8"])
im = ax.imshow(class_numeric, cmap=cmap, aspect="auto")
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("Attention Head Classification Map")
ax.set_xticks(range(n_heads))
ax.set_yticks(range(0, n_layers, 2))

# Legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#b87333", label="Sink"),
    Patch(facecolor="#4a7c74", label="Previous-token"),
    Patch(facecolor="#5a7a3d", label="Self-attention"),
    Patch(facecolor="#6b7091", label="Uniform"),
    Patch(facecolor="#d4c8a8", label="Mixed"),
]
ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
fig.tight_layout()
show(fig, filename="02-head-classification-map.png")

# %% [markdown]
# ## Part 8: Head Ablation — Which Heads Matter?
#
# Now the question that connects theory to practice: if we zero out a single
# head, how much does the model's performance change?
#
# We will measure perplexity — how surprised the model is by a set of sentences.
# Lower perplexity means better predictions. If ablating a head raises perplexity
# sharply, that head is load-bearing. If it barely changes, the head is redundant.

# %%
# Evaluation sentences for perplexity
EVAL_TEXTS = [
    "The research team published their findings in a prestigious journal last month.",
    "Small language models are increasingly used in edge computing applications.",
    "The attention mechanism allows each token to gather information from other positions.",
    "Training on high quality data often matters more than training on more data.",
    "The capital of France is Paris and the capital of Japan is Tokyo.",
]

# Compute baseline perplexity (no ablation)
print("Computing baseline perplexity...")
baseline_ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS, device=device)
console.print(f"\n[bold]Baseline perplexity:[/bold] {baseline_ppl:.2f}")

# %%
# Ablate each head and measure the impact
# For time: ablate every head in a subset of layers, or all heads if fast enough
N_LAYERS_TO_ABLATE = n_layers if not is_ci() else 4  # Reduce in CI
layers_to_ablate = list(range(N_LAYERS_TO_ABLATE))

HEAD_DIM = 128  # Qwen3-0.6B head dimension

print(
    f"\nAblating {N_LAYERS_TO_ABLATE} layers x {n_heads} heads"
    f" = {N_LAYERS_TO_ABLATE * n_heads} experiments..."
)

ablation_results = np.zeros((N_LAYERS_TO_ABLATE, n_heads))

from tqdm import tqdm

for i, layer_idx in enumerate(tqdm(layers_to_ablate, desc="Ablating layers")):
    for head_idx in range(n_heads):
        with ablate_head(model, layer_idx, head_idx, head_dim=HEAD_DIM):
            ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS, device=device)
        ablation_results[i, head_idx] = ppl - baseline_ppl

# %%
# Visualize the ablation impact
fig, ax = plt.subplots(figsize=(14, max(6, N_LAYERS_TO_ABLATE * 0.3)))
im = ax.imshow(
    ablation_results,
    cmap="RdYlGn_r",
    aspect="auto",
    vmin=0,
    vmax=max(1.0, np.percentile(ablation_results, 95)),
)
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("Ablation Impact: Perplexity Increase When Head Is Zeroed")
ax.set_xticks(range(n_heads))
ax.set_yticks(range(N_LAYERS_TO_ABLATE))
ax.set_yticklabels([str(idx) for idx in layers_to_ablate])
plt.colorbar(im, ax=ax, label="Perplexity increase (higher = more important)")
fig.tight_layout()
show(fig, filename="02-ablation-impact.png")

# %%
# Find the most and least important heads
top_k = 5
flat_results = [
    (layers_to_ablate[i], j, ablation_results[i, j])
    for i in range(N_LAYERS_TO_ABLATE)
    for j in range(n_heads)
]
flat_results.sort(key=lambda x: -x[2])

table = Table(title=f"Top {top_k} Most Important Heads (highest perplexity increase)")
table.add_column("Layer", justify="right")
table.add_column("Head", justify="right")
table.add_column("PPL Change", justify="right")
for layer, head, delta in flat_results[:top_k]:
    table.add_row(str(layer), str(head), f"+{delta:.3f}")
console.print(table)

table = Table(title=f"Top {top_k} Least Important Heads (smallest perplexity change)")
table.add_column("Layer", justify="right")
table.add_column("Head", justify="right")
table.add_column("PPL Change", justify="right")
for layer, head, delta in flat_results[-top_k:]:
    sign = "+" if delta >= 0 else ""
    table.add_row(str(layer), str(head), f"{sign}{delta:.3f}")
console.print(table)

# %%
# How many heads can you remove with minimal damage?
threshold = 0.1  # Less than 0.1 PPL increase is negligible
negligible = sum(1 for _, _, d in flat_results if d < threshold)
console.print(
    f"\n[bold]{negligible} of {N_LAYERS_TO_ABLATE * n_heads} heads[/bold] "
    f"({negligible / (N_LAYERS_TO_ABLATE * n_heads):.0%}) cause less than "
    f"{threshold} perplexity increase when ablated."
)
console.print("This is the empirical basis for head pruning: most heads are redundant.")

# %% [markdown]
# ## Key Takeaways
#
# **What you measured:**
#
# 1. **Attention sinks are real.** Many heads across all layers send attention
#    to the first token, regardless of content. This is a learned "no-op" —
#    when a head has nothing useful to attend to, the first token acts as a
#    safe default. (Xiao et al., 2023)
#
# 2. **Previous-token heads concentrate in early layers.** These heads implement
#    a simple "look at what came before me" pattern. They provide positional
#    information that later layers build on — a key component of the induction
#    circuit described by Olsson et al. (2022).
#
# 3. **GQA pairs share structure.** Heads that share key-value projections produce
#    similar but not identical patterns. The query projection gives each head
#    room to specialize, even with shared KV.
#
# 4. **Most heads are dispensable.** Ablating the majority of heads causes
#    negligible perplexity increase. A few critical heads carry most of the
#    model's function. This is why structured pruning works — you can remove
#    heads and save compute without meaningful quality loss.
#
# ## Artifacts
#
# Check your `outputs/` directory for:
# - `02-attention-landscape.png` — three-panel overview of all 448 heads
# - `02-attention-head-types.png` — examples of sink, previous-token, and broad heads
# - `02-gqa-pairs.png` — GQA paired heads showing shared structure
# - `02-attention-grid.png` — layer-by-layer evolution of attention patterns
# - `02-head-classification-map.png` — pattern type for every head
# - `02-ablation-impact.png` — which heads matter most
#
# ## References
#
# - Olsson et al., "In-context Learning and Induction Heads" (2022)
# - Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
# - Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
# - Zheng et al., "Attention Heads of Large Language Models: A Survey" (2025)
