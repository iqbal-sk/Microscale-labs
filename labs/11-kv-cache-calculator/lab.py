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
# # Lab 11: KV Cache Budget Calculator
#
# **Act VIII — Serving the Model** | CPU only | ~60 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Understand** why the KV cache exists — it stores past key and value
#    tensors so the model doesn't recompute them on every new token
# 2. **Build** a calculator that predicts exact KV cache memory from any
#    model's config (layers, KV heads, head_dim, dtype)
# 3. **Explore** how sequence length, batch size, and quantization affect
#    the memory budget
# 4. **Compare** KV cache sizes across the models you've studied in these labs
# 5. **See** why KV cache — not model weights — is often the limiting
#    factor for serving multiple concurrent users
#
# ---
#
# ### The idea
#
# During autoregressive generation, every new token attends to all
# previous tokens. Without caching, this means recomputing all K and V
# projections from scratch on every step — quadratic in sequence length.
#
# The **KV cache** stores the K and V tensors for all past positions,
# so only the new token's K and V need to be computed. This makes
# generation linear in sequence length — but at the cost of memory
# that grows with every token.
#
# The formula:
#
# $$\text{KV cache (bytes)} = 2 \times L \times H_{kv} \times d_h
# \times S \times B \times b$$
#
# Where: L = layers, H_kv = KV heads, d_h = head dim, S = sequence length,
# B = batch size, b = bytes per element (2 for fp16, 1 for int8).

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
            "git+https://github.com/iqbal-sk/Microscale-labs.git",
        ]
    )
    # Force Python to see the newly installed package (important on Colab)
    import importlib

    importlib.invalidate_caches()
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
# ## 2. The KV Cache Formula
#
# For each layer, we store:
# - **K cache**: (batch, num_kv_heads, seq_len, head_dim)
# - **V cache**: (batch, num_kv_heads, seq_len, head_dim)
#
# Total across all layers:
#
# ```
# kv_bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element
#            ↑                                                       ↑
#         K + V                                              fp16=2, int8=1
# ```

# %%


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,  # fp16
) -> int:
    """Calculate KV cache size in bytes."""
    return (
        2  # K + V
        * n_layers
        * n_kv_heads
        * head_dim
        * seq_len
        * batch_size
        * bytes_per_element
    )


def format_bytes(n: int) -> str:
    """Human-readable byte size."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


# %% [markdown]
# ---
# ## 3. Models We Know
#
# Let's compute KV cache for all the models we've studied.

# %%
MODELS = {
    "SmolLM2-360M": {
        "layers": 32,
        "kv_heads": 3,
        "head_dim": 96,
        "q_heads": 10,
        "hidden": 960,
        "params": "362M",
    },
    "Qwen3-0.6B": {
        "layers": 28,
        "kv_heads": 8,
        "head_dim": 128,
        "q_heads": 16,
        "hidden": 1024,
        "params": "596M",
    },
    "SmolLM3-3B": {
        "layers": 36,
        "kv_heads": 4,
        "head_dim": 128,
        "q_heads": 16,
        "hidden": 2048,
        "params": "3.08B",
    },
    "Phi-4-mini": {
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "q_heads": 24,
        "hidden": 3072,
        "params": "3.84B",
    },
    "Llama-3.2-3B": {
        "layers": 28,
        "kv_heads": 8,
        "head_dim": 128,
        "q_heads": 24,
        "hidden": 3072,
        "params": "3.21B",
    },
}

# Compute KV cache at different sequence lengths
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

table = Table(title="KV Cache Size (FP16, batch=1)")
table.add_column("Model", style="bold")
table.add_column("Params")
table.add_column("GQA Ratio")
for s in [512, 2048, 8192, 32768]:
    table.add_column(f"seq={s:,}", justify="right")

for name, cfg in MODELS.items():
    row = [name, cfg["params"], f"{cfg['q_heads']}:{cfg['kv_heads']}"]
    for s in [512, 2048, 8192, 32768]:
        size = kv_cache_bytes(
            cfg["layers"],
            cfg["kv_heads"],
            cfg["head_dim"],
            s,
        )
        row.append(format_bytes(size))
    table.add_row(*row)

console.print(table)

# %% [markdown]
# Notice how the KV cache grows linearly with sequence length.
# At seq=32768, even the small Qwen3-0.6B needs ~1.8 GB just for KV cache —
# comparable to the model weights themselves.

# %% [markdown]
# ---
# ## 4. Visualize: KV Cache vs Sequence Length

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: KV cache size vs sequence length
ax = axes[0]
colors = ["#b87333", "#4a7c74", "#8b3a3a", "#5a7a3d", "#1a1f3a"]
for (name, cfg), color in zip(MODELS.items(), colors):
    sizes_mb = [
        kv_cache_bytes(cfg["layers"], cfg["kv_heads"], cfg["head_dim"], s) / 1e6
        for s in SEQ_LENGTHS
    ]
    ax.plot(SEQ_LENGTHS, sizes_mb, marker="o", label=name, color=color, linewidth=2, markersize=5)

ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("KV Cache Size (MB)")
ax.set_title("KV Cache Growth with Sequence Length\n(FP16, batch=1)")
ax.set_xscale("log", base=2)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: KV cache as fraction of model weights
ax = axes[1]
# Approximate model weight sizes in MB (fp16)
weight_sizes = {
    "SmolLM2-360M": 724,
    "Qwen3-0.6B": 1192,
    "SmolLM3-3B": 6160,
    "Phi-4-mini": 7680,
    "Llama-3.2-3B": 6420,
}

for (name, cfg), color in zip(MODELS.items(), colors):
    fractions = [
        kv_cache_bytes(cfg["layers"], cfg["kv_heads"], cfg["head_dim"], s)
        / (weight_sizes[name] * 1e6)
        * 100
        for s in SEQ_LENGTHS
    ]
    ax.plot(SEQ_LENGTHS, fractions, marker="s", label=name, color=color, linewidth=2, markersize=5)

ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("KV Cache as % of Model Weights")
ax.set_title("When Does KV Cache Exceed Model Size?")
ax.set_xscale("log", base=2)
ax.axhline(100, color="#6b7091", linestyle="--", alpha=0.5, label="100% (cache = model)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "KV Cache Memory Analysis",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="11-kv-cache-growth.png")

# %% [markdown]
# **Key observation:** For small models with aggressive GQA (like
# SmolLM2-360M with 3 KV heads), the KV cache stays small even at
# long sequences. For models with more KV heads, it can exceed the
# model weight size at long contexts.

# %% [markdown]
# ---
# ## 5. Predicted vs Actual — Load the Real Model
#
# The formula gives us a prediction. Let's load Qwen3-0.6B, run actual
# inference at different sequence lengths, and measure the real KV cache
# tensors to see how accurate our formula is.

# %%
import torch
from transformers import AutoModelForCausalLM

dev = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading Qwen3-0.6B on {dev}...")
kv_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float16,
).to(dev)
kv_model.eval()

# %%
test_seq_lengths = [64, 256, 512, 1024, 2048]
predicted_mb = []
actual_mb = []

for seq_len in test_seq_lengths:
    input_ids = torch.randint(0, 1000, (1, seq_len), device=dev)

    with torch.no_grad():
        outputs = kv_model(input_ids, use_cache=True)

    # Measure actual KV cache from the DynamicCache object
    pkv = outputs.past_key_values
    actual_bytes = 0
    for layer in pkv.layers:
        actual_bytes += layer.keys.nelement() * layer.keys.element_size()
        actual_bytes += layer.values.nelement() * layer.values.element_size()

    # Our formula's prediction
    pred_bytes = kv_cache_bytes(28, 8, 128, seq_len, 1, 2)

    predicted_mb.append(pred_bytes / 1e6)
    actual_mb.append(actual_bytes / 1e6)

    console.print(
        f"  seq={seq_len:5d}  "
        f"predicted={pred_bytes / 1e6:7.1f} MB  "
        f"actual={actual_bytes / 1e6:7.1f} MB  "
        f"match={'[green]YES[/]' if pred_bytes == actual_bytes else '[red]NO[/]'}"
    )

# Show the first layer's cache shape for understanding
layer0 = pkv.layers[0]
console.print(
    f"\n  Cache shape per layer: K={list(layer0.keys.shape)}  V={list(layer0.values.shape)}"
)
console.print(
    f"  (batch={layer0.keys.shape[0]},"
    f" kv_heads={layer0.keys.shape[1]},"
    f" seq_len={layer0.keys.shape[2]},"
    f" head_dim={layer0.keys.shape[3]})"
)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    test_seq_lengths,
    predicted_mb,
    "o--",
    color="#b87333",
    linewidth=2,
    markersize=8,
    label="Predicted (formula)",
)
ax.plot(
    test_seq_lengths,
    actual_mb,
    "s-",
    color="#4a7c74",
    linewidth=2,
    markersize=8,
    label="Actual (measured)",
)
ax.set_xlabel("Sequence Length")
ax.set_ylabel("KV Cache Size (MB)")
ax.set_title(
    "Predicted vs Actual KV Cache — Qwen3-0.6B",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
show(fig, filename="11-kv-predicted-vs-actual.png")

# %% [markdown]
# The formula matches the actual cache tensors exactly — no overhead,
# no approximation. This is because the KV cache is simply a stack of
# tensors with a known shape. The "overhead" in real serving comes from
# memory fragmentation, CUDA context, attention score buffers, and
# framework bookkeeping — not from the cache itself.

# %%
# Clean up to free memory before batch size analysis
del kv_model, outputs, pkv
torch.mps.empty_cache() if torch.backends.mps.is_available() else None

# %% [markdown]
# ---
# ## 6. The Batch Size Problem
#
# For serving, the real constraint is **concurrent users**. Each user
# needs their own KV cache. Let's see how batch size affects total
# memory.

# %%
model_name = "Qwen3-0.6B"
cfg = MODELS[model_name]
model_weight_mb = weight_sizes[model_name]

SEQ = 2048  # Typical conversation length

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
kv_mb = [
    kv_cache_bytes(
        cfg["layers"],
        cfg["kv_heads"],
        cfg["head_dim"],
        SEQ,
        batch_size=b,
    )
    / 1e6
    for b in batch_sizes
]
total_mb = [model_weight_mb + kv for kv in kv_mb]

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    range(len(batch_sizes)),
    [model_weight_mb] * len(batch_sizes),
    color="#4a7c74",
    label="Model weights",
    width=0.6,
)
ax.bar(
    range(len(batch_sizes)),
    kv_mb,
    bottom=[model_weight_mb] * len(batch_sizes),
    color="#b87333",
    label="KV cache",
    width=0.6,
)

for i, (total, kv) in enumerate(zip(total_mb, kv_mb)):
    ax.text(i, total + 50, f"{total:.0f}\nMB", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(range(len(batch_sizes)))
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.set_xlabel("Batch Size (concurrent users)")
ax.set_ylabel("Total GPU Memory (MB)")
ax.set_title(
    f"Memory Budget: {model_name} at seq_len={SEQ:,}\n"
    f"Model weights: {model_weight_mb:,} MB (fixed)",
    fontsize=12,
    fontweight="bold",
)
ax.legend()
ax.grid(True, alpha=0.2, axis="y")

# Add memory limit lines
for mem_gb, label in [(8, "8 GB GPU"), (16, "16 GB GPU"), (24, "24 GB GPU")]:
    ax.axhline(mem_gb * 1024, color="#8b3a3a", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(
        len(batch_sizes) - 0.5, mem_gb * 1024 + 100, label, fontsize=8, color="#8b3a3a", ha="right"
    )

ax.set_ylim(0, max(total_mb) * 1.3)
fig.tight_layout()
show(fig, filename="11-kv-cache-batch-scaling.png")

# %% [markdown]
# The model weights are a **fixed cost** — the same regardless of how
# many users you serve. The KV cache is a **per-user cost** that scales
# linearly with batch size. This is why PagedAttention (vLLM) matters:
# it manages KV cache memory like a virtual memory system, allowing
# efficient sharing and allocation.

# %% [markdown]
# ---
# ## 7. GQA Saves KV Memory
#
# Grouped Query Attention reduces the KV cache by using fewer KV heads.
# Let's quantify the savings.

# %%
table = Table(title="GQA Impact on KV Cache (seq=2048, batch=1, FP16)")
table.add_column("Model", style="bold")
table.add_column("Q Heads")
table.add_column("KV Heads")
table.add_column("GQA Ratio")
table.add_column("KV Cache", justify="right")
table.add_column("If MHA (no GQA)", justify="right")
table.add_column("Savings", justify="right", style="green")

for name, cfg in MODELS.items():
    actual = kv_cache_bytes(
        cfg["layers"],
        cfg["kv_heads"],
        cfg["head_dim"],
        2048,
    )
    # What if we had full MHA (kv_heads = q_heads)?
    mha = kv_cache_bytes(
        cfg["layers"],
        cfg["q_heads"],
        cfg["head_dim"],
        2048,
    )
    savings = (1 - actual / mha) * 100

    table.add_row(
        name,
        str(cfg["q_heads"]),
        str(cfg["kv_heads"]),
        f"{cfg['q_heads']}:{cfg['kv_heads']}",
        format_bytes(actual),
        format_bytes(mha),
        f"{savings:.0f}%",
    )

console.print(table)

# %% [markdown]
# SmolLM3-3B saves **75%** of KV cache memory thanks to its aggressive
# 4:1 GQA ratio. This directly translates to serving 4x more concurrent
# users on the same GPU.

# %% [markdown]
# ---
# ## 8. Quantized KV Cache
#
# You can also quantize the KV cache itself (separate from weight
# quantization). Using int8 instead of fp16 halves the cache size.

# %%
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(MODELS))
width = 0.25

for i, (dtype_name, bpe) in enumerate([("FP16", 2), ("INT8", 1), ("FP8", 1)]):
    sizes = [
        kv_cache_bytes(
            cfg["layers"],
            cfg["kv_heads"],
            cfg["head_dim"],
            4096,
            bytes_per_element=bpe,
        )
        / 1e6
        for cfg in MODELS.values()
    ]
    ax.bar(
        x + i * width,
        sizes,
        width,
        label=f"KV cache ({dtype_name})",
        color=["#4a7c74", "#b87333", "#5a7a3d"][i],
        edgecolor="#1a1f3a",
        linewidth=0.5,
    )

ax.set_xticks(x + width)
ax.set_xticklabels(MODELS.keys(), fontsize=10)
ax.set_ylabel("KV Cache Size (MB)")
ax.set_title(
    "KV Cache Size with Different Precision\n(seq=4096, batch=1)",
    fontsize=12,
    fontweight="bold",
)
ax.legend()
ax.grid(True, alpha=0.2, axis="y")
fig.tight_layout()
show(fig, filename="11-kv-cache-quantization.png")

# %% [markdown]
# ---
# ## 9. Interactive Calculator
#
# Use this function to compute KV cache for any model config.

# %%


def kv_cache_report(
    model_name: str,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype: str = "fp16",
):
    """Print a KV cache memory report."""
    bpe = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "fp8": 1}[dtype]
    cache = kv_cache_bytes(
        n_layers,
        n_kv_heads,
        head_dim,
        seq_len,
        batch_size,
        bpe,
    )

    table = Table(title=f"KV Cache Report: {model_name}")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Layers", str(n_layers))
    table.add_row("KV Heads", str(n_kv_heads))
    table.add_row("Head Dim", str(head_dim))
    table.add_row("Sequence Length", f"{seq_len:,}")
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Dtype", dtype)
    table.add_row("KV Cache Size", f"[bold]{format_bytes(cache)}[/bold]")
    table.add_row(
        "Per Token",
        format_bytes(cache // seq_len),
    )
    console.print(table)


# Example: Qwen3-0.6B at different configs
kv_cache_report("Qwen3-0.6B", 28, 8, 128, seq_len=4096, batch_size=1)
kv_cache_report("Qwen3-0.6B", 28, 8, 128, seq_len=4096, batch_size=32)

# %% [markdown]
# ---
# ## What You Learned
#
# | Concept | Detail |
# |---|---|
# | KV cache formula | 2 x L x H_kv x d_h x S x B x b |
# | Linear in seq_len | Double the context = double the cache |
# | Linear in batch | Each user needs their own cache |
# | GQA reduces cache | Fewer KV heads = proportionally less memory |
# | KV quantization | INT8 cache halves memory vs FP16 |
# | Model weights are fixed | KV cache is the per-user variable cost |
#
# ### Artifacts in `outputs/`
#
# | File | What it shows |
# |------|---------------|
# | `11-kv-cache-growth.png` | Cache size vs sequence length |
# | `11-kv-cache-batch-scaling.png` | Memory budget with batch size |
# | `11-kv-cache-quantization.png` | FP16 vs INT8 vs FP8 cache |
#
# ### References
#
# - Kwon et al., "Efficient Memory Management for Large Language
#   Model Serving with PagedAttention" (vLLM, 2023)
# - Pope et al., "Efficiently Scaling Transformer Inference" (2022)
