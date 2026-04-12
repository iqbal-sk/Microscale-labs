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
# # Lab 10: The Roofline Lab
#
# **Act VIII — Serving the Model** | GPU required | ~60–90 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Measure** your GPU's real memory bandwidth — not the spec-sheet
#    number, the sustained throughput under load
# 2. **Measure** your GPU's real compute throughput — TFLOPS on a
#    dense matrix multiplication
# 3. **Build a roofline chart** from your measurements — the fundamental
#    performance model for any computation
# 4. **Plot** where LLM inference sits on the roofline at different
#    batch sizes
# 5. **See** why single-token decode is always bandwidth-bound and
#    why batching pushes you toward the compute ceiling
#
# ---
#
# ### The idea
#
# Every GPU has two limits: how fast it can move data (bandwidth) and
# how fast it can compute (FLOPS). The roofline model plots both as
# a function of arithmetic intensity (FLOPs per byte transferred).
#
# For LLM inference at batch=1, arithmetic intensity is ~1 FLOP/byte
# for fp16 weights — firmly in the bandwidth-bound regime. This is why
# your GPU's spec-sheet TFLOPS number is irrelevant for single-user
# inference: the bottleneck is memory bandwidth.

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

from microscale import apply_style, device_summary, get_torch_device, show

apply_style()
device = get_torch_device()
print(device_summary())

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

if device.type == "cpu":
    console.print(
        "[bold red]This lab requires a GPU (CUDA or MPS).[/]\n"
        "  Results on CPU will not be meaningful.\n"
    )

# %% [markdown]
# ---
# ## 2. Measure Memory Bandwidth
#
# We allocate a large tensor and copy it — measuring how many GB/s
# the GPU can sustain for pure data movement.

# %%
# Allocate ~256 MB for the bandwidth test
BW_SIZE = 64 * 1024 * 1024  # 64M floats = 256 MB
N_BW_ITERS = 20


def measure_bandwidth(dev, size=BW_SIZE, n_iters=N_BW_ITERS):
    """Measure sustained memory bandwidth in GB/s."""
    x = torch.randn(size, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    bytes_per_copy = x.nelement() * x.element_size() * 2  # read + write

    # Warmup
    for _ in range(3):
        y.copy_(x)

    # Synchronize
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y.copy_(x)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    gb_per_sec = (bytes_per_copy * n_iters) / elapsed / 1e9
    return gb_per_sec


bw_gb_s = measure_bandwidth(device)
console.print(f"\n  [bold]Measured memory bandwidth: {bw_gb_s:.1f} GB/s[/bold]")

# %% [markdown]
# ---
# ## 3. Measure Compute Throughput
#
# We run a large dense matrix multiplication and measure TFLOPS.
# FLOPs for an NxN matmul = 2 * N^3 (multiply-accumulate).

# %%
MATMUL_N = 4096
N_MATMUL_ITERS = 10


def measure_compute(dev, n=MATMUL_N, n_iters=N_MATMUL_ITERS):
    """Measure sustained compute in TFLOPS (fp16 or fp32)."""
    # Use fp16 on CUDA (tensor cores), fp32 on MPS
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    flops_per_matmul = 2 * n**3  # 2*N^3 for matrix multiply

    # Warmup
    for _ in range(3):
        torch.mm(a, b)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.mm(a, b)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = (flops_per_matmul * n_iters) / elapsed / 1e12
    return tflops, dtype


compute_tflops, compute_dtype = measure_compute(device)
console.print(f"  [bold]Measured compute: {compute_tflops:.2f} TFLOPS ({compute_dtype})[/bold]")

# %% [markdown]
# ---
# ## 4. Build the Roofline
#
# The roofline model:
#
# $$\text{Performance} = \min(\text{Peak FLOPS},\;
# \text{Peak BW} \times \text{Arithmetic Intensity})$$
#
# The **ridge point** — where bandwidth and compute ceilings meet —
# tells you the minimum arithmetic intensity needed to be compute-bound:
#
# $$\text{AI}_{\text{ridge}} = \frac{\text{Peak FLOPS}}{\text{Peak BW}}$$

# %%
# Ridge point
ridge_point = compute_tflops * 1e12 / (bw_gb_s * 1e9)

table = Table(title="Your GPU Profile")
table.add_column("Metric", style="bold")
table.add_column("Value", justify="right")
table.add_row("Memory Bandwidth", f"{bw_gb_s:.1f} GB/s")
table.add_row("Compute Throughput", f"{compute_tflops:.2f} TFLOPS")
table.add_row("Ridge Point", f"{ridge_point:.1f} FLOP/byte")
table.add_row("Device", str(device))
console.print(table)

# %% [markdown]
# ---
# ## 5. Where Does LLM Inference Sit?
#
# For a model with P parameters stored in `b_w` bytes per weight:
#
# | Scenario | FLOPs | Bytes Read | Arithmetic Intensity |
# |----------|-------|-----------|---------------------|
# | Batch=1 decode (fp16) | 2P | 2P | **1.0 FLOP/byte** |
# | Batch=1 decode (4-bit) | 2P | 0.5P | **4.0 FLOP/byte** |
# | Batch=B decode (fp16) | 2PB | 2P | **B FLOP/byte** |
#
# At batch=1 with fp16 weights, AI = 1.0 — deep in the bandwidth-bound
# region (ridge point is typically 10-50).

# %%
# Model operating points
model_params = 596e6  # Qwen3-0.6B

operating_points = [
    ("Batch=1, FP16", 1.0, "#8b3a3a"),
    ("Batch=1, 4-bit", 4.0, "#b87333"),
    ("Batch=4, FP16", 4.0, "#4a7c74"),
    ("Batch=16, FP16", 16.0, "#5a7a3d"),
    ("Batch=64, FP16", 64.0, "#1a1f3a"),
]

# Compute expected performance at each operating point
for name, ai, color in operating_points:
    bw_limited = bw_gb_s * 1e9 * ai  # FLOP/s
    compute_limited = compute_tflops * 1e12  # FLOP/s
    actual = min(bw_limited, compute_limited)
    regime = "bandwidth-bound" if ai < ridge_point else "compute-bound"
    tok_per_sec = actual / (2 * model_params)

    console.print(
        f"  {name:20s}  AI={ai:5.1f}  "
        f"{actual / 1e12:.2f} TFLOP/s  "
        f"~{tok_per_sec:.0f} tok/s  "
        f"[dim]({regime})[/dim]"
    )

# %% [markdown]
# ---
# ## 6. The Roofline Chart
#
# This is the most important visualization in GPU computing.

# %%
fig, ax = plt.subplots(figsize=(14, 8))

# Arithmetic intensity range (log scale)
ai_range = np.logspace(-1, 3, 500)

# Bandwidth ceiling: performance = BW * AI
bw_ceiling = bw_gb_s * ai_range  # in GFLOP/s (GB/s * FLOP/byte)

# Compute ceiling: flat line at peak TFLOPS
compute_ceiling = np.full_like(ai_range, compute_tflops * 1e3)  # GFLOP/s

# Roofline = min of both
roofline = np.minimum(bw_ceiling, compute_ceiling)

# Plot roofline
ax.loglog(ai_range, roofline, color="#1a1f3a", linewidth=3, zorder=3)
ax.fill_between(
    ai_range,
    roofline,
    0.01,
    alpha=0.05,
    color="#1a1f3a",
)

# Mark the ridge point
ax.axvline(
    ridge_point,
    color="#6b7091",
    linestyle="--",
    alpha=0.5,
    label=f"Ridge point ({ridge_point:.1f} FLOP/byte)",
)

# Plot operating points
for name, ai, color in operating_points:
    bw_limited = bw_gb_s * ai  # GFLOP/s
    perf = min(bw_limited, compute_tflops * 1e3)
    ax.scatter(
        ai,
        perf,
        s=150,
        color=color,
        zorder=5,
        edgecolors="#1a1f3a",
        linewidth=1.5,
    )
    ax.annotate(
        name,
        (ai, perf),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        color=color,
        fontweight="bold",
    )

# Labels
ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
ax.set_title(
    f"Roofline Model — Your {device.type.upper()} GPU\n"
    f"BW: {bw_gb_s:.0f} GB/s  |  Compute: {compute_tflops:.1f} TFLOPS",
    fontsize=13,
    fontweight="bold",
)

# Region labels
ax.text(
    0.15,
    compute_tflops * 300,
    "BANDWIDTH\nBOUND",
    fontsize=14,
    color="#8b3a3a",
    alpha=0.3,
    fontweight="bold",
    ha="center",
)
ax.text(
    ridge_point * 10,
    compute_tflops * 300,
    "COMPUTE\nBOUND",
    fontsize=14,
    color="#4a7c74",
    alpha=0.3,
    fontweight="bold",
    ha="center",
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, which="both")
ax.set_xlim(0.1, 1000)
ax.set_ylim(1, compute_tflops * 2e3)

fig.tight_layout()
show(fig, filename="10-roofline-chart.png")

# %% [markdown]
# **Reading the chart:**
# - The **diagonal line** (left of ridge) is the bandwidth ceiling —
#   performance scales linearly with arithmetic intensity
# - The **flat line** (right of ridge) is the compute ceiling —
#   you've saturated the ALUs
# - Points **left of the ridge** are bandwidth-bound (most LLM inference)
# - Points **right of the ridge** are compute-bound (batched prefill,
#   training)
#
# Batch=1 decode sits at AI=1.0 — far left, deep in bandwidth territory.
# This is why a GPU with 2x more TFLOPS but the same bandwidth gives
# you almost no speedup for chat inference.

# %% [markdown]
# ---
# ## 7. Practical Implications
#
# Let's compute expected tokens/second for Qwen3-0.6B on your hardware.

# %%
table = Table(title=f"Expected Throughput — Qwen3-0.6B on {device.type.upper()}")
table.add_column("Config", style="bold")
table.add_column("AI (FLOP/byte)", justify="right")
table.add_column("Regime")
table.add_column("Tokens/sec", justify="right")
table.add_column("Bottleneck")

configs = [
    ("FP16, batch=1", 2, 1.0),
    ("4-bit, batch=1", 0.5, 4.0),
    ("FP16, batch=8", 2, 8.0),
    ("FP16, batch=32", 2, 32.0),
]

for name, bytes_per_w, ai in configs:
    bw_perf = bw_gb_s * 1e9 * ai
    compute_perf = compute_tflops * 1e12
    actual = min(bw_perf, compute_perf)
    regime = "BW-bound" if ai < ridge_point else "Compute-bound"

    # Tokens/sec: total FLOP/s / FLOPs-per-token
    # FLOPs per token ≈ 2 * P (forward pass)
    tok_s = actual / (2 * model_params)

    # What's the bottleneck?
    if ai < ridge_point:
        bottleneck = f"Memory ({bw_gb_s:.0f} GB/s)"
    else:
        bottleneck = f"Compute ({compute_tflops:.1f} TFLOPS)"

    table.add_row(name, f"{ai:.1f}", regime, f"{tok_s:.0f}", bottleneck)

console.print(table)

# %% [markdown]
# **Key insight:** Going from FP16 to 4-bit quantization gives a ~4x
# speedup for batch=1 decode — not because you're computing faster, but
# because you're reading 4x fewer bytes from memory. The compute is the
# same; the bottleneck (bandwidth) is what improved.
#
# Increasing batch size also helps because you reuse the same weights
# for multiple tokens — amortizing the memory read across the batch.

# %% [markdown]
# ---
# ## 8. Bandwidth Efficiency
#
# How close are you to your GPU's theoretical maximum?

# %%
# Look up known spec-sheet values for common GPUs
KNOWN_SPECS = {
    "mps": {
        "M1": 68,
        "M1 Pro": 200,
        "M1 Max": 400,
        "M2": 100,
        "M2 Pro": 200,
        "M2 Max": 400,
        "M3": 100,
        "M3 Pro": 150,
        "M3 Max": 400,
        "M4": 120,
        "M4 Pro": 273,
        "M4 Max": 546,
    },
    "cuda": {
        "T4": 300,
        "A10G": 600,
        "A100": 2039,
        "RTX 3060": 360,
        "RTX 3090": 936,
        "RTX 4060": 272,
        "RTX 4070": 504,
        "RTX 4090": 1008,
    },
}

console.print(f"\n  Your measured bandwidth: [bold]{bw_gb_s:.1f} GB/s[/bold]")
console.print("  Typical spec-sheet values for reference:")

specs = KNOWN_SPECS.get(device.type, {})
if specs:
    for name, bw in specs.items():
        console.print(f"    {name:15s}: {bw:,} GB/s")
    console.print(
        "\n  [dim]Your measured value should be 75-92% of your GPU's spec-sheet bandwidth.[/dim]"
    )

# %% [markdown]
# ---
# ## What You Learned
#
# | Concept | Your measurement |
# |---|---|
# | Memory bandwidth | How fast your GPU reads weights |
# | Compute throughput | Your GPU's TFLOPS ceiling |
# | Ridge point | Where BW-bound meets compute-bound |
# | Batch=1 decode | Firmly bandwidth-bound (AI ≈ 1) |
# | 4-bit quantization | 4x bandwidth savings → 4x faster decode |
# | Batching | Amortizes weight reads → pushes toward compute |
#
# ### Artifacts in `outputs/`
#
# | File | What it shows |
# |------|---------------|
# | `10-roofline-chart.png` | Your GPU's roofline with model operating points |
#
# ### References
#
# - Williams, Waterman & Patterson, "Roofline: An Insightful Visual
#   Performance Model" (2009)
# - The roofline model is the standard framework for understanding
#   GPU performance in ML inference and HPC
