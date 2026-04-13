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
# # Lab 09: Quantize It Yourself
#
# **Act VII — Packing for Travel** | CPU only | ~90 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Implement naive 4-bit quantization** — uniform bins across [min, max]
# 2. **Implement NF4** — quantile-based bins that are optimal for
#    Gaussian-distributed weights (the QLoRA approach)
# 3. **Implement K-quant style** — sub-block scales that adapt to local
#    weight distributions (the llama.cpp approach)
# 4. **Measure the error** — MSE, SQNR, and max error for each scheme
# 5. **See why the weight distribution matters** — Gaussian weights waste
#    half the bins in naive quantization, but NF4 uses them all
# 6. **Connect error to perplexity** — understand why better quantization
#    = better model quality at the same bit width
#
# ---
#
# ### The idea
#
# A 4-bit weight can only take 16 possible values. The question is:
# *which* 16 values? Naive quantization spaces them evenly across the
# range. NF4 spaces them at the quantiles of a normal distribution.
# K-quant adapts the spacing per 32-weight sub-block.
#
# You are about to implement all three and measure the difference on
# a real weight tensor from Qwen3-0.6B.

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
# ## 2. Load a Real Weight Tensor
#
# We extract a single weight matrix from Qwen3-0.6B — the query
# projection from layer 5. This is a [2048, 1024] matrix with
# 2,097,152 parameters. In fp16 it takes 4 MB.

# %%
import json
import struct

from huggingface_hub import hf_hub_download

# Download just the safetensors file
print("Downloading Qwen3-0.6B weights...")
model_path = hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors")

# Read the safetensors header to find our tensor's byte offsets
with open(model_path, "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_size))

# Extract layer 5 q_proj
tensor_name = "model.layers.5.self_attn.q_proj.weight"
tensor_info = header[tensor_name]
start, end = tensor_info["data_offsets"]

with open(model_path, "rb") as f:
    f.seek(8 + header_size + start)
    raw_bytes = f.read(end - start)

# Convert BF16 bytes to float32 numpy array
import torch

bf16_tensor = torch.frombuffer(
    bytearray(raw_bytes),
    dtype=torch.bfloat16,
).reshape(tensor_info["shape"])
weights = bf16_tensor.float().numpy()

console.print(f"\n  Tensor: {tensor_name}")
console.print(f"  Shape: {weights.shape}")
console.print(f"  Parameters: {weights.size:,}")
console.print(f"  FP16 size: {weights.size * 2 / 1e6:.1f} MB")
console.print(f"  Range: [{weights.min():.4f}, {weights.max():.4f}]")
console.print(f"  Mean: {weights.mean():.6f}, Std: {weights.std():.4f}")

# %% [markdown]
# ---
# ## 3. The Weight Distribution
#
# Before quantizing, let's look at what we're working with. Neural
# network weights are approximately Gaussian — this fact is what
# makes NF4 work.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram
ax = axes[0]
flat = weights.flatten()
ax.hist(flat, bins=200, color="#b87333", alpha=0.8, edgecolor="none", density=True)

# Overlay a Gaussian fit
mu, sigma = flat.mean(), flat.std()
x_gauss = np.linspace(flat.min(), flat.max(), 200)
gauss = np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
ax.plot(x_gauss, gauss, color="#1a1f3a", linewidth=2, label="Gaussian fit")
ax.set_xlabel("Weight Value")
ax.set_ylabel("Density")
ax.set_title("Weight Distribution (Layer 5 Q-Projection)")
ax.legend()
ax.grid(True, alpha=0.3)

# QQ plot
ax = axes[1]
sorted_weights = np.sort(flat)
n = len(sorted_weights)
theoretical = np.sort(np.random.randn(n) * sigma + mu)
sample_idx = np.linspace(0, n - 1, 1000).astype(int)
ax.scatter(
    theoretical[sample_idx],
    sorted_weights[sample_idx],
    s=3,
    color="#4a7c74",
    alpha=0.5,
)
lims = [min(theoretical.min(), sorted_weights.min()), max(theoretical.max(), sorted_weights.max())]
ax.plot(lims, lims, color="#8b3a3a", linewidth=1.5, linestyle="--", label="Perfect Gaussian")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Observed Quantiles")
ax.set_title("Q-Q Plot: How Gaussian Are the Weights?")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Real Transformer Weights Are Approximately Gaussian",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="09-weight-distribution.png")

# %% [markdown]
# The histogram fits a Gaussian closely, and the Q-Q plot follows the
# diagonal. This is why NF4 works: it places its 16 bin centers at the
# quantiles of N(0,1), giving equal probability mass to each bin —
# information-theoretically optimal for this distribution.

# %% [markdown]
# ---
# ## 4. Scheme 1 — Naive Uniform 4-bit
#
# The simplest approach: divide [min, max] into 16 equal bins.
#
# **Problem**: Most weights cluster near zero, so most bins are wasted
# on the tails where almost no weights live.

# %%
from microscale.quantize import (
    NF4_VALUES,
    bits_per_weight,
    dequantize_naive_4bit,
    dequantize_nf4,
    dequantize_q4k,
    quantization_error,
    quantize_naive_4bit,
    quantize_nf4,
    quantize_q4k,
)

# Quantize
indices_naive, scale, zp = quantize_naive_4bit(weights.flatten())
deq_naive = dequantize_naive_4bit(indices_naive, scale, zp).reshape(weights.shape)
err_naive = quantization_error(weights, deq_naive)

console.print("\n  [bold]Naive 4-bit[/bold]")
console.print(f"  SQNR: {err_naive['sqnr_db']:.1f} dB")
console.print(f"  RMSE: {err_naive['rmse']:.6f}")
console.print(f"  Max error: {err_naive['max_error']:.6f}")

# %% [markdown]
# ---
# ## 5. Scheme 2 — NF4 (Normal Float 4)
#
# From QLoRA (Dettmers et al., 2023). Uses 16 quantile values of N(0,1)
# as bin centers, with per-block absmax scaling.
#
# The 16 NF4 values are:
# ```
# -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
#  0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0
# ```
#
# Notice the asymmetry: bins are denser near zero (where most weights
# live) and sparser in the tails (where few weights live).

# %%
indices_nf4, scales_nf4 = quantize_nf4(weights.flatten(), block_size=64)
deq_nf4 = dequantize_nf4(
    indices_nf4,
    scales_nf4,
    block_size=64,
    original_shape=weights.shape,
)
err_nf4 = quantization_error(weights, deq_nf4)

console.print("\n  [bold]NF4 (block_size=64)[/bold]")
console.print(f"  SQNR: {err_nf4['sqnr_db']:.1f} dB")
console.print(f"  RMSE: {err_nf4['rmse']:.6f}")
console.print(f"  Max error: {err_nf4['max_error']:.6f}")

# %% [markdown]
# ---
# ## 6. Scheme 3 — K-quant Q4_K Style
#
# From llama.cpp. Uses a two-level scaling hierarchy:
# - **Super-block** (256 weights): one fp16 scale
# - **Sub-block** (32 weights): one 6-bit scale relative to the super-block
#
# This gives each group of 32 weights its own scale, adapting to local
# weight distributions.

# %%
indices_q4k, sub_scales, sub_mins, super_scales = quantize_q4k(
    weights.flatten(),
    super_block_size=256,
    sub_block_size=32,
)
deq_q4k = dequantize_q4k(
    indices_q4k,
    sub_scales,
    sub_mins,
    original_shape=weights.shape,
)
err_q4k = quantization_error(weights, deq_q4k)

console.print("\n  [bold]Q4_K style (sub_block=32)[/bold]")
console.print(f"  SQNR: {err_q4k['sqnr_db']:.1f} dB")
console.print(f"  RMSE: {err_q4k['rmse']:.6f}")
console.print(f"  Max error: {err_q4k['max_error']:.6f}")

# %% [markdown]
# ---
# ## 7. Side-by-Side Comparison

# %%
table = Table(title="Quantization Error Comparison")
table.add_column("Method", style="bold")
table.add_column("SQNR (dB)", justify="right")
table.add_column("RMSE", justify="right")
table.add_column("Max Error", justify="right")
table.add_column("Bits/Weight", justify="right")

methods = [
    ("FP16 (baseline)", {"sqnr_db": float("inf"), "rmse": 0.0, "max_error": 0.0}, 16.0),
    ("Naive 4-bit", err_naive, bits_per_weight("naive", weights.size)),
    ("NF4 (block=64)", err_nf4, bits_per_weight("nf4", weights.size, block_size=64)),
    ("Q4_K (sub=32)", err_q4k, bits_per_weight("q4k", weights.size)),
]

for name, err, bpw in methods:
    sqnr = f"{err['sqnr_db']:.1f}" if err["sqnr_db"] < 1e6 else "inf"
    table.add_row(
        name,
        sqnr,
        f"{err['rmse']:.6f}",
        f"{err['max_error']:.6f}",
        f"{bpw:.2f}",
    )
console.print(table)

# %% [markdown]
# ---
# ## 8. Visualize the Differences

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Bin placement comparison
ax = axes[0, 0]
naive_bins = np.linspace(weights.min(), weights.max(), 16)
nf4_bins = NF4_VALUES * weights.std()  # scaled to match weight range
ax.hist(flat, bins=200, color="#d4c8a8", alpha=0.6, density=True, label="Weight distribution")
for i, b in enumerate(naive_bins):
    ax.axvline(b, color="#8b3a3a", alpha=0.5, linewidth=1, label="Naive bins" if i == 0 else None)
for i, b in enumerate(nf4_bins):
    ax.axvline(b, color="#4a7c74", alpha=0.7, linewidth=1.5, label="NF4 bins" if i == 0 else None)
ax.set_xlabel("Weight Value")
ax.set_title("Bin Placement: Naive vs NF4")
ax.legend(fontsize=9)
ax.set_xlim(weights.min() * 1.1, weights.max() * 1.1)

# Panel 2: Error distribution per method
ax = axes[0, 1]
err_flat_naive = (weights - deq_naive).flatten()
err_flat_nf4 = (weights - deq_nf4).flatten()
err_flat_q4k = (weights - deq_q4k).flatten()
ax.hist(
    err_flat_naive,
    bins=100,
    alpha=0.5,
    color="#8b3a3a",
    label=f"Naive (RMSE={err_naive['rmse']:.5f})",
    density=True,
)
ax.hist(
    err_flat_nf4,
    bins=100,
    alpha=0.5,
    color="#4a7c74",
    label=f"NF4 (RMSE={err_nf4['rmse']:.5f})",
    density=True,
)
ax.hist(
    err_flat_q4k,
    bins=100,
    alpha=0.5,
    color="#b87333",
    label=f"Q4_K (RMSE={err_q4k['rmse']:.5f})",
    density=True,
)
ax.set_xlabel("Quantization Error")
ax.set_title("Error Distribution by Method")
ax.legend(fontsize=9)

# Panel 3: SQNR bar chart
ax = axes[1, 0]
names = ["Naive 4-bit", "NF4", "Q4_K"]
sqnrs = [err_naive["sqnr_db"], err_nf4["sqnr_db"], err_q4k["sqnr_db"]]
colors_bar = ["#8b3a3a", "#4a7c74", "#b87333"]
bars = ax.bar(names, sqnrs, color=colors_bar, edgecolor="#1a1f3a", linewidth=1.5)
for bar, val in zip(bars, sqnrs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f} dB",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylabel("SQNR (dB) — higher is better")
ax.set_title("Signal-to-Quantization-Noise Ratio")

# Panel 4: Scatter — original vs dequantized (sample)
ax = axes[1, 1]
sample_idx = np.random.choice(weights.size, 2000, replace=False)
flat_orig = weights.flatten()
ax.scatter(
    flat_orig[sample_idx],
    deq_nf4.flatten()[sample_idx],
    s=2,
    alpha=0.3,
    color="#4a7c74",
    label="NF4",
)
ax.scatter(
    flat_orig[sample_idx],
    deq_naive.flatten()[sample_idx],
    s=2,
    alpha=0.3,
    color="#8b3a3a",
    label="Naive",
)
lim = max(abs(flat_orig.min()), abs(flat_orig.max())) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=1)
ax.set_xlabel("Original Weight")
ax.set_ylabel("Dequantized Weight")
ax.set_title("Original vs Dequantized (2000 samples)")
ax.legend(fontsize=9)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

fig.suptitle(
    "Three Approaches to 4-bit Quantization",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="09-quantization-comparison.png")

# %% [markdown]
# **What the plots show:**
#
# - **Bin placement** (top-left): Naive bins are evenly spaced, wasting
#   resolution in the tails. NF4 bins cluster near zero where the density
#   is highest.
# - **Error distribution** (top-right): NF4 and Q4_K produce tighter
#   error distributions than naive.
# - **SQNR** (bottom-left): Higher is better. NF4 should beat naive by
#   several dB because its bins match the weight distribution.
# - **Scatter** (bottom-right): Points closer to the diagonal = less
#   quantization error. NF4 follows the line more tightly.

# %% [markdown]
# ---
# ## 9. Compression Summary

# %%
original_bytes = weights.size * 2  # fp16
naive_bytes = weights.size // 2 + 8  # 4-bit packed + scale/zp
nf4_bytes = weights.size // 2 + len(scales_nf4) * 4  # 4-bit + fp32 scales
q4k_bytes = weights.size // 2 + len(sub_scales) * 4 + len(sub_mins) * 4 + len(super_scales) * 4

table = Table(title="Compression Results")
table.add_column("Format", style="bold")
table.add_column("Size", justify="right")
table.add_column("Compression", justify="right")
table.add_column("SQNR", justify="right")

for name, size, sqnr in [
    ("FP16 (original)", original_bytes, "inf"),
    ("Naive 4-bit", naive_bytes, f"{err_naive['sqnr_db']:.1f}"),
    ("NF4", nf4_bytes, f"{err_nf4['sqnr_db']:.1f}"),
    ("Q4_K", q4k_bytes, f"{err_q4k['sqnr_db']:.1f}"),
]:
    table.add_row(
        name,
        f"{size / 1e6:.2f} MB",
        f"{original_bytes / size:.1f}x",
        f"{sqnr} dB",
    )
console.print(table)

# %% [markdown]
# ---
# ## 10. Save the Quantized Tensors

# %%
from microscale.viz import _output_dir

output_path = _output_dir()
np.save(output_path / "09-weights-fp16.npy", weights)
np.save(output_path / "09-weights-naive4.npy", deq_naive)
np.save(output_path / "09-weights-nf4.npy", deq_nf4)
np.save(output_path / "09-weights-q4k.npy", deq_q4k)

console.print(f"\n  Saved quantized tensors to {output_path}")

# %% [markdown]
# ---
# ## What You Learned
#
# | Method | How it works | Best for |
# |---|---|---|
# | Naive 4-bit | Uniform bins across [min, max] | Simplicity |
# | NF4 | Gaussian quantile bins + per-block scale | Gaussian weights (most LLMs) |
# | Q4_K | Per-32-weight sub-block scales | Varying local distributions |
#
# | Concept | Detail |
# |---|---|
# | Weights are Gaussian | Validated with histogram and Q-Q plot |
# | Naive wastes bins | Uniform spacing ignores the distribution |
# | NF4 matches the data | Quantile bins = optimal for Gaussian |
# | Sub-block scales adapt | Local precision where weights vary |
# | SQNR measures quality | Higher dB = less quantization noise |
#
# ### Artifacts in `outputs/`
#
# | File | What it is |
# |------|-----------|
# | `09-weight-distribution.png` | Histogram + Q-Q plot |
# | `09-quantization-comparison.png` | 4-panel error comparison |
# | `09-weights-*.npy` | Original + 3 dequantized tensors |
#
# ### References
#
# - Dettmers et al., "QLoRA" (2023, arXiv:2305.14314)
# - llama.cpp K-quant implementation (ggml-quants.c)
# - The complete implementation lives in `microscale/quantize.py`
