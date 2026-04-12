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
# # Lab 04: Model Autopsy
#
# **Act III — The Current Champions** | CPU only | ~45–60 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Parse** the safetensors binary format — read model architecture from
#    a 35KB header without downloading gigabytes of weights
# 2. **Detect** architecture choices from tensor names alone: GQA ratio,
#    tied embeddings, SwiGLU, fused projections
# 3. **Compare** four production SLMs side by side — two sub-1B and two 3B+ —
#    and understand why they made different design trade-offs
# 4. **Measure** where parameters actually go — and see how the vocabulary
#    tax shrinks as models get larger
#
# ---
#
# ### The idea
#
# Every model on HuggingFace is a collection of named tensors. The names
# and shapes tell you everything about the architecture — how many layers,
# how many heads, whether embeddings are tied, what kind of FFN it uses.
# You do not need to load a single weight to know all of this.
#
# Today you will dissect four models — from 360M to 3.8B parameters —
# and build a comparison table from their anatomy.

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
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/user/Microscale.git"]
    )
    import microscale

from microscale import apply_style, device_summary, show

apply_style()
print(device_summary())

# %%
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# %% [markdown]
# ---
# ## 2. The Safetensors Format
#
# A safetensors file starts with an 8-byte integer (the header size),
# followed by a JSON dictionary describing every tensor. The actual
# weight data comes after the header.
#
# ```
# [8 bytes: header_size][header_size bytes: JSON][... weight data ...]
# ```
#
# The JSON maps tensor names to their dtype, shape, and byte offsets.
# By reading only the first ~35KB, we can learn the full architecture.

# %%
import json
import struct

import requests
from huggingface_hub import get_token, hf_hub_url


def read_safetensors_header(repo_id: str, filename: str) -> dict:
    """Read just the header of a safetensors file via HTTP Range request."""
    url = hf_hub_url(repo_id, filename)
    auth = {}
    token = get_token()
    if token:
        auth["Authorization"] = f"Bearer {token}"

    # 8-byte little-endian u64: header size
    resp = requests.get(url, headers={**auth, "Range": "bytes=0-7"}, timeout=30)
    resp.raise_for_status()
    header_size = struct.unpack("<Q", resp.content)[0]

    # Read the JSON header
    resp = requests.get(
        url,
        headers={**auth, "Range": f"bytes=8-{8 + header_size - 1}"},
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.content)


# %% [markdown]
# Let's read Qwen3-0.6B's header and see what's inside:

# %%
header = read_safetensors_header("Qwen/Qwen3-0.6B", "model.safetensors")

# Filter out metadata
tensors_raw = {k: v for k, v in header.items() if k != "__metadata__"}
console.print(f"  Found [bold]{len(tensors_raw)}[/bold] tensors\n")

# Show first 10
table = Table(title="Qwen3-0.6B — First 10 Tensors")
table.add_column("Name", style="bold")
table.add_column("Dtype")
table.add_column("Shape", justify="right")
table.add_column("Parameters", justify="right")

for i, (name, info) in enumerate(sorted(tensors_raw.items())):
    if i >= 10:
        break
    numel = 1
    for dim in info["shape"]:
        numel *= dim
    table.add_row(name, info["dtype"], str(info["shape"]), f"{numel:,}")

console.print(table)

# %% [markdown]
# Each tensor name encodes its location in the architecture:
# - `model.layers.{N}.self_attn.q_proj.weight` — layer N's query projection
# - `model.layers.{N}.mlp.gate_proj.weight` — layer N's SwiGLU gate
# - `model.embed_tokens.weight` — the input embedding table
#
# From names and shapes alone, you can reconstruct the full architecture.

# %% [markdown]
# ---
# ## 3. Automated Architecture Detection
#
# The `microscale.autopsy` module automates this: it reads the header,
# parses tensor names, and infers the architecture.

# %%
from microscale.autopsy import (
    analyze_architecture,
    format_params,
    parse_safetensors_header_remote,
)

MODELS = {
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B",
    "Phi-4-mini": "microsoft/Phi-4-mini-instruct",
}

anatomies = {}
for name, repo in MODELS.items():
    console.print(f"  Analyzing [bold]{name}[/bold]...")
    tensors = parse_safetensors_header_remote(repo)
    anatomies[name] = analyze_architecture(tensors, repo=repo)

console.print("  Done.\n")

# %% [markdown]
# ---
# ## 4. Side-by-Side Architecture Comparison

# %%
table = Table(title="Architecture Comparison")
table.add_column("Property", style="bold")
for name in MODELS:
    table.add_column(name, justify="right")

rows = [
    ("Total Parameters", lambda a: format_params(a.total_params)),
    ("Layers", lambda a: str(a.num_layers)),
    ("Hidden Size", lambda a: f"{a.hidden_size:,}"),
    ("Intermediate Size", lambda a: f"{a.intermediate_size:,}"),
    ("Vocabulary", lambda a: f"{a.vocab_size:,}"),
    ("Query Heads", lambda a: str(a.num_q_heads)),
    ("KV Heads", lambda a: str(a.num_kv_heads)),
    ("GQA Ratio", lambda a: f"{a.num_q_heads}:{a.num_kv_heads}"),
    ("Head Dim", lambda a: str(a.head_dim)),
    ("FFN Type", lambda a: a.ffn_type),
    ("Tied Embeddings", lambda a: "Yes" if a.has_tied_embeddings else "No"),
    ("QK-Norm", lambda a: "Yes" if a.has_qk_norm else "No"),
    ("Fused QKV", lambda a: "Yes" if a.has_fused_qkv else "No"),
    ("Tensor Count", lambda a: str(len(a.tensors))),
]

for label, fn in rows:
    table.add_row(label, *[fn(anatomies[n]) for n in MODELS])

console.print(table)

# %% [markdown]
# **What jumps out:**
#
# - **SmolLM2-360M** uses the smallest vocabulary (49K) and the most
#   aggressive GQA (10:3). At 360M, every parameter counts.
# - **Qwen3-0.6B** has a huge vocabulary (152K) relative to its size,
#   and is the only model with QK-Norm.
# - **SmolLM3-3B** has the largest FFN ratio (intermediate = 5.4x hidden)
#   — most of its parameters are in the feed-forward layers.
# - **Phi-4-mini** fuses Q/K/V into a single `qkv_proj` matrix and
#   gate/up into `gate_up_proj`, halving its tensor count to 194.

# %% [markdown]
# ---
# ## 5. Where Do the Parameters Go?
#
# This is the key question for small models. Let's visualize the
# parameter breakdown.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
colors = ["#b87333", "#4a7c74", "#5a7a3d", "#6b7091", "#d4c8a8"]
components = ["Embedding", "Attention", "FFN", "Norm", "Other"]

for ax, name in zip(axes.flat, MODELS):
    a = anatomies[name]
    sizes = [
        a.embedding_params,
        a.attention_params,
        a.ffn_params,
        a.norm_params,
        a.other_params,
    ]
    # Filter out zeros
    nonzero = [(s, c, col) for s, c, col in zip(sizes, components, colors) if s > 0]
    vals = [x[0] for x in nonzero]
    labels = [f"{x[1]}\n{format_params(x[0])}" for x in nonzero]
    cols = [x[2] for x in nonzero]

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        colors=cols,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
        pctdistance=0.6,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title(f"{name}\n{format_params(a.total_params)} total", fontsize=11)

fig.suptitle(
    "Parameter Distribution — Where Do the Parameters Go?",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="04-parameter-distribution.png")

# %% [markdown]
# **The vocabulary tax — and how it changes with scale:**
#
# Compare the two smallest models:
# - **SmolLM2-360M** keeps its vocab small (49K) to spend only ~13% on
#   embeddings. Smart at this scale — every parameter matters.
# - **Qwen3-0.6B** uses a much larger vocab (152K) and pays ~26% for it.
#   That is a quarter of the model doing nothing but mapping token IDs.
#
# At 3B+, the tax fades:
# - **SmolLM3-3B** spends only 8.5% on embeddings despite 128K vocab,
#   freeing budget for an enormous 79% FFN ratio.
# - **Phi-4-mini** has the largest vocab (200K) but at 3.8B it is only 16%.
#
# The pattern: vocabulary is a fixed cost. As models grow, it becomes
# a smaller fraction. Small models must choose their vocab size carefully.

# %% [markdown]
# ---
# ## 6. Parameter Efficiency Comparison
#
# How do these models compare in terms of "useful" parameters (non-embedding)?

# %%
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(MODELS.keys())
x = np.arange(len(model_names))
width = 0.18

component_data = {
    "Embedding": [anatomies[n].embedding_params / 1e9 for n in model_names],
    "Attention": [anatomies[n].attention_params / 1e9 for n in model_names],
    "FFN": [anatomies[n].ffn_params / 1e9 for n in model_names],
}

for i, (comp, vals) in enumerate(component_data.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=comp, color=colors[i])
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}B",
                ha="center",
                va="bottom",
                fontsize=8,
            )

ax.set_ylabel("Parameters (billions)")
ax.set_title(
    "Parameter Budget by Component",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend()
ax.set_ylim(0, max(max(v) for v in component_data.values()) * 1.3)
fig.tight_layout()
show(fig, filename="04-parameter-budget.png")

# %% [markdown]
# ---
# ## 7. Architecture Design Choices
#
# Let's map the design space these models occupy.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: GQA ratio vs model size
ax = axes[0]
for name in model_names:
    a = anatomies[name]
    ratio = a.num_q_heads / a.num_kv_heads if a.num_kv_heads else 1
    ax.scatter(
        a.total_params / 1e9,
        ratio,
        s=200,
        zorder=5,
        color="#b87333",
    )
    ax.annotate(
        name,
        (a.total_params / 1e9, ratio),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=10,
    )
ax.set_xlabel("Total Parameters (B)")
ax.set_ylabel("GQA Ratio (Q heads / KV heads)")
ax.set_title("GQA Compression vs Model Size")
ax.grid(True, alpha=0.3)

# Plot 2: FFN ratio (intermediate / hidden) vs model size
ax = axes[1]
for name in model_names:
    a = anatomies[name]
    ffn_ratio = a.intermediate_size / a.hidden_size if a.hidden_size else 0
    ax.scatter(
        a.total_params / 1e9,
        ffn_ratio,
        s=200,
        zorder=5,
        color="#4a7c74",
    )
    ax.annotate(
        name,
        (a.total_params / 1e9, ffn_ratio),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=10,
    )
ax.set_xlabel("Total Parameters (B)")
ax.set_ylabel("FFN Expansion Ratio")
ax.set_title("FFN Width vs Model Size")
ax.axhline(y=4.0, color="#6b7091", linestyle="--", alpha=0.5, label="4x (typical)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Design Space — Architecture Choices",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="04-design-space.png")

# %% [markdown]
# **Design choices visible from the data:**
#
# - **GQA ratio increases with size:** Qwen3-0.6B uses 2:1 (conservative),
#   Phi-4-mini uses 3:1 (moderate), SmolLM3-3B uses 4:1 (aggressive).
#   Larger models can afford more KV compression.
#
# - **FFN width varies dramatically:** SmolLM3-3B uses 5.4x expansion
#   (vs the standard 4x). This trades attention capacity for FFN capacity —
#   a bet that knowledge storage (in FFN) matters more than relationship
#   modeling (in attention) at this scale.

# %% [markdown]
# ---
# ## 8. Tensor-Level View
#
# Let's look at the raw tensor inventory for each model.

# %%
for name in model_names:
    a = anatomies[name]

    # Count tensors per component
    attn_count = sum(
        1
        for t in a.tensors
        if any(k in t.name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"])
    )
    ffn_count = sum(
        1
        for t in a.tensors
        if any(k in t.name for k in ["gate_proj", "up_proj", "down_proj", "gate_up_proj"])
    )
    norm_count = sum(
        1
        for t in a.tensors
        if "norm" in t.name and "q_norm" not in t.name and "k_norm" not in t.name
    )
    qknorm_count = sum(1 for t in a.tensors if "q_norm" in t.name or "k_norm" in t.name)

    table = Table(title=f"{name} — Tensor Inventory")
    table.add_column("Component", style="bold")
    table.add_column("Tensors", justify="right")
    table.add_column("Per Layer", justify="right")
    table.add_row("Attention projections", str(attn_count), str(attn_count // a.num_layers))
    table.add_row("FFN projections", str(ffn_count), str(ffn_count // a.num_layers))
    norms_per_layer = str(norm_count // a.num_layers) if a.num_layers else "0"
    table.add_row("Layer norms", str(norm_count), norms_per_layer)
    if qknorm_count:
        table.add_row("QK-Norm", str(qknorm_count), str(qknorm_count // a.num_layers))
    table.add_row("Embedding", "1", "—")
    if any("lm_head" in t.name for t in a.tensors):
        table.add_row("LM Head", "1 (tied)" if a.has_tied_embeddings else "1", "—")
    table.add_row("[bold]Total[/bold]", f"[bold]{len(a.tensors)}[/bold]", "")
    console.print(table)
    console.print()

# %% [markdown]
# **Why Phi-4 has half the tensors:**
#
# Phi-4-mini fuses Q+K+V into one `qkv_proj` matrix and gate+up into
# `gate_up_proj`. This means 2 tensors per attention (qkv + o) instead
# of 4 (q + k + v + o), and 2 tensors per FFN (gate_up + down) instead
# of 3 (gate + up + down). The total parameter count is the same — it is
# a storage optimization, not a capacity change.

# %% [markdown]
# ---
# ## What You Learned
#
# | Finding | Evidence |
# |---|---|
# | Headers reveal architecture | 35KB tells you everything |
# | Vocab tax shrinks with scale | 26% at 596M vs 8.5% at 3B |
# | FFN ratios vary dramatically | 2.7x to 5.4x expansion |
# | GQA gets more aggressive | 2:1 to 4:1 compression |
# | Fused projections halve tensor count | 194 vs 311 |
#
# ### Artifacts in `outputs/`
#
# | File | What it shows |
# |------|---------------|
# | `04-parameter-distribution.png` | Pie charts of param breakdown |
# | `04-parameter-budget.png` | Bar chart comparing components |
# | `04-design-space.png` | GQA ratio and FFN width scatter |
#
# ### References
#
# - HuggingFace safetensors format specification
# - Ainslie et al., "GQA" (2023)
# - Shazeer, "GLU Variants Improve Transformer" (2020)
