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
# # Lab 03: Build a Transformer Block from Raw Ops
#
# **Act II — Inside the Machine** | CPU or Apple Silicon | ~90–120 minutes
#
# ---
#
# ### What you will learn
#
# By the end of this lab you will be able to:
#
# 1. **Implement RMSNorm** — the normalization layer used in modern transformers
# 2. **Implement Rotary Position Embeddings (RoPE)** — how transformers encode
#    position without adding position vectors
# 3. **Implement Grouped Query Attention (GQA)** — including the QK-Norm step
#    that is specific to Qwen3
# 4. **Implement SwiGLU** — the gated feed-forward network used in most
#    current models
# 5. **Assemble a full decoder layer** and load real weights from Qwen3-0.6B
# 6. **Verify your code** produces output identical to HuggingFace's
#    implementation — `torch.allclose` with zero tolerance
#
# ---
#
# ### Why this matters
#
# Reading about attention is one thing. Implementing it — the actual matrix
# multiplications, the RoPE rotations, the causal mask — and watching your
# implementation produce the exact same numbers as a production model? That
# is how you go from "I understand the diagram" to "I understand the machine."
#
# Every component in this lab is something you will encounter in model
# architecture papers, quantization code, and inference engines. After
# today, none of it will be a black box.

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

from microscale import apply_style, device_summary, get_torch_device, show

apply_style()
device = get_torch_device()
print(device_summary())

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()

# %% [markdown]
# ---
# ## 2. Load the Reference Model
#
# We will use HuggingFace's Qwen3-0.6B as ground truth. At each step, we
# build a component from scratch and verify it matches the reference.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3-0.6B (reference model)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ref_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float32,
    attn_implementation="eager",
).to(device)
ref_model.eval()

# Print layer 0 architecture
ref_layer = ref_model.model.layers[0]
table = Table(title="Qwen3-0.6B Layer 0 — Weight Shapes")
table.add_column("Parameter", style="bold")
table.add_column("Shape", justify="right")
table.add_column("Elements", justify="right")
for name, param in ref_layer.named_parameters():
    table.add_row(name, str(list(param.shape)), f"{param.numel():,}")
console.print(table)

# %% [markdown]
# ---
# ## 3. Component 1 — RMSNorm
#
# **What it does:** Normalizes activations by their root-mean-square, then
# scales by a learned weight. Unlike LayerNorm, it skips the mean-centering
# step, which makes it cheaper and works just as well in practice.
#
# **Formula:**
# $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$
#
# **Key detail:** The computation is done in float32 for numerical stability,
# even when the model runs in bfloat16.

# %%


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# %% [markdown]
# Let's verify against HuggingFace's implementation:

# %%
# Copy weights from the reference
our_norm = RMSNorm(1024, eps=1e-6).to(device)
our_norm.weight.data.copy_(ref_layer.input_layernorm.weight.data)

# Test with a random input
test_input = torch.randn(1, 8, 1024, device=device)

with torch.no_grad():
    ref_out = ref_layer.input_layernorm(test_input)
    our_out = our_norm(test_input)

diff = (ref_out - our_out).abs().max().item()
console.print(
    f"  RMSNorm max diff: [bold]{diff:.2e}[/bold]  "
    + ("[green]PASS[/]" if diff < 1e-6 else "[red]FAIL[/]")
)

# %% [markdown]
# ---
# ## 4. Component 2 — Rotary Position Embeddings (RoPE)
#
# **What it does:** Encodes position information by rotating query and key
# vectors. Unlike learned position embeddings, RoPE naturally handles
# sequences longer than seen during training.
#
# **How it works:** Each pair of dimensions in the head vector gets rotated
# by an angle proportional to the position. Lower-frequency dimensions
# rotate slowly (capturing long-range patterns), higher-frequency dimensions
# rotate quickly (capturing local patterns).
#
# **The math:**
# - Compute inverse frequencies: $\theta_i = \text{base}^{-2i/d}$
# - For each position $p$: angles $= p \cdot \theta_i$
# - Rotate: $x' = x \cos\theta + \text{rotate\_half}(x) \sin\theta$

# %%


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
    rope_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=rope_device) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=rope_device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split tensor in half along last dim, negate second half, swap."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# %% [markdown]
# Let's visualize what the RoPE frequency spectrum looks like:

# %%
cos_table, sin_table = compute_rope_frequencies(128, 64, theta=1e6, rope_device=device)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(cos_table.cpu().numpy(), cmap="RdBu", aspect="auto")
axes[0].set_title("Cosine Table (positions x dimensions)")
axes[0].set_xlabel("Dimension")
axes[0].set_ylabel("Position")

axes[1].imshow(sin_table.cpu().numpy(), cmap="RdBu", aspect="auto")
axes[1].set_title("Sine Table (positions x dimensions)")
axes[1].set_xlabel("Dimension")
axes[1].set_ylabel("Position")

fig.suptitle(
    "RoPE Frequency Tables — Lower dims rotate faster, higher dims rotate slower",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="03-rope-frequencies.png")

# %% [markdown]
# Notice the wave patterns: the leftmost dimensions (low indices) oscillate
# rapidly across positions, while the rightmost dimensions barely change.
# This multi-frequency encoding lets the model simultaneously capture
# fine-grained local position and coarse long-range position.

# %% [markdown]
# ---
# ## 5. Component 3 — Grouped Query Attention with QK-Norm
#
# This is the core of the transformer. Here is the exact sequence of
# operations we need to implement:
#
# 1. Project input to Q, K, V using separate linear layers
# 2. Reshape to heads: Q gets 16 heads, K and V get 8 heads (GQA)
# 3. **QK-Norm**: Apply RMSNorm per-head to Q and K (Qwen3-specific)
# 4. Transpose to (batch, heads, seq, dim)
# 5. Apply RoPE to Q and K
# 6. Repeat K, V heads to match Q (8 → 16 by duplicating each)
# 7. Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
# 8. Apply causal mask (no attending to future tokens)
# 9. Reshape and project output
#
# **Qwen3 quirks to get right:**
# - QK-Norm happens AFTER reshaping to heads but BEFORE RoPE
# - The Q projection goes from 1024 → 2048 (upsamples! because 16 heads × 128 dim > 1024)
# - No bias on any projection

# %%


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads."""
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5

        # Projections (no bias)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK-Norm (per-head RMSNorm)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, hidden_states, cos, sin):
        batch, seq_len, _ = hidden_states.shape

        # Step 1: Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Step 2: Reshape to heads
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Step 3: QK-Norm (per-head, BEFORE transpose, BEFORE RoPE)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Step 4: Transpose to (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Step 5: Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 6: Repeat KV for GQA
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Step 7: Attention scores
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # Step 8: Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

        # Step 9: Softmax (in float32)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)

        # Step 10: Weighted values
        attn_output = torch.matmul(attn_weights, v)

        # Step 11: Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


# %% [markdown]
# ---
# ## 6. Component 4 — SwiGLU Feed-Forward Network
#
# **What it does:** Expands the hidden representation, applies a gated
# nonlinearity, then projects back down.
#
# **Formula:** `output = down_proj(silu(gate_proj(x)) * up_proj(x))`
#
# The "gate" path controls how much of the "up" path passes through.
# SiLU (also called Swish) is the smooth activation: `silu(x) = x * sigmoid(x)`.

# %%


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# %% [markdown]
# Let's verify the FFN against the reference:

# %%
our_ffn = SwiGLUFFN(1024, 3072).to(device)
our_ffn.gate_proj.weight.data.copy_(ref_layer.mlp.gate_proj.weight.data)
our_ffn.up_proj.weight.data.copy_(ref_layer.mlp.up_proj.weight.data)
our_ffn.down_proj.weight.data.copy_(ref_layer.mlp.down_proj.weight.data)

test_input = torch.randn(1, 8, 1024, device=device)
with torch.no_grad():
    ref_ffn_out = ref_layer.mlp(test_input)
    our_ffn_out = our_ffn(test_input)

diff = (ref_ffn_out - our_ffn_out).abs().max().item()
console.print(
    f"  SwiGLU FFN max diff: [bold]{diff:.2e}[/bold]  "
    + ("[green]PASS[/]" if diff < 1e-6 else "[red]FAIL[/]")
)

# %% [markdown]
# ---
# ## 7. Assembly — The Full Decoder Layer
#
# Now we put it all together. A transformer decoder layer is:
#
# ```
# residual = x
# x = RMSNorm(x)            ← input layernorm
# x = Attention(x)           ← grouped query attention
# x = residual + x           ← first residual connection
#
# residual = x
# x = RMSNorm(x)            ← post-attention layernorm
# x = SwiGLU(x)             ← feed-forward network
# x = residual + x           ← second residual connection
# ```

# %%


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=3072,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, hidden_states, cos, sin):
        # Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# %% [markdown]
# ---
# ## 8. The Moment of Truth
#
# We will now:
# 1. Create our TransformerBlock
# 2. Copy every weight from Qwen3-0.6B layer 0
# 3. Feed the same input to both
# 4. Check if the outputs match

# %%
# Build our block
our_block = TransformerBlock().to(device)

# Copy all weights from layer 0
hf_layer = ref_model.model.layers[0]

weight_map = {
    "self_attn.q_proj.weight": hf_layer.self_attn.q_proj.weight,
    "self_attn.k_proj.weight": hf_layer.self_attn.k_proj.weight,
    "self_attn.v_proj.weight": hf_layer.self_attn.v_proj.weight,
    "self_attn.o_proj.weight": hf_layer.self_attn.o_proj.weight,
    "self_attn.q_norm.weight": hf_layer.self_attn.q_norm.weight,
    "self_attn.k_norm.weight": hf_layer.self_attn.k_norm.weight,
    "input_layernorm.weight": hf_layer.input_layernorm.weight,
    "post_attention_layernorm.weight": hf_layer.post_attention_layernorm.weight,
    "mlp.gate_proj.weight": hf_layer.mlp.gate_proj.weight,
    "mlp.up_proj.weight": hf_layer.mlp.up_proj.weight,
    "mlp.down_proj.weight": hf_layer.mlp.down_proj.weight,
}

for name, hf_param in weight_map.items():
    our_param = dict(our_block.named_parameters())[name]
    our_param.data.copy_(hf_param.data)

console.print(f"  Copied {len(weight_map)} weight tensors from layer 0")

# %%
# Capture layer 0's input from the reference model
layer0_inputs = []


def capture_input(module, args, kwargs):
    layer0_inputs.append(args[0].clone())


hook = ref_model.model.layers[0].register_forward_pre_hook(capture_input, with_kwargs=True)

text = "The scientist observed the experiment carefully"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    ref_outputs = ref_model(**inputs, output_hidden_states=True)

hook.remove()

# The input to layer 0 is the embedding output
hf_input = layer0_inputs[0]  # (1, seq_len, 1024)
# The output of layer 0 is hidden_states[1] (index 0 = embeddings)
hf_layer0_output = ref_outputs.hidden_states[1]

seq_len = hf_input.shape[1]
console.print(f"  Input shape: {list(hf_input.shape)}")
console.print(f"  Sequence length: {seq_len} tokens")

# %%
# Run through our block
cos, sin = compute_rope_frequencies(128, seq_len, theta=1e6, rope_device=device)

with torch.no_grad():
    our_output = our_block(hf_input, cos, sin)

# %% [markdown]
# ### The Comparison

# %%
max_diff = (our_output - hf_layer0_output).abs().max().item()
mean_diff = (our_output - hf_layer0_output).abs().mean().item()
matches = torch.allclose(our_output, hf_layer0_output, atol=1e-5)

table = Table(title="Verification: Our Block vs HuggingFace Layer 0")
table.add_column("Metric", style="bold")
table.add_column("Value", justify="right")
table.add_row("Max absolute difference", f"{max_diff:.2e}")
table.add_row("Mean absolute difference", f"{mean_diff:.2e}")
table.add_row(
    "torch.allclose(atol=1e-5)",
    "[green bold]PASS[/]" if matches else "[red bold]FAIL[/]",
)
console.print(table)

if matches:
    console.print(
        "\n  [green bold]Your from-scratch transformer block"
        " produces identical output to HuggingFace.[/]\n"
    )

# %% [markdown]
# ---
# ## 9. Visualize the Internals
#
# Now that our block works, let's look inside it. We can visualize
# the weight distributions and the attention patterns from our own code.

# %%
# Weight distribution across components
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
weight_groups = [
    ("Q Projection", our_block.self_attn.q_proj.weight),
    ("K Projection", our_block.self_attn.k_proj.weight),
    ("V Projection", our_block.self_attn.v_proj.weight),
    ("Gate Projection", our_block.mlp.gate_proj.weight),
    ("Up Projection", our_block.mlp.up_proj.weight),
    ("Down Projection", our_block.mlp.down_proj.weight),
]

for ax, (name, weight) in zip(axes.flat, weight_groups):
    data = weight.data.cpu().float().numpy().flatten()
    ax.hist(data, bins=100, color="#b87333", alpha=0.8, edgecolor="none")
    ax.set_title(f"{name}\n{list(weight.shape)}", fontsize=10)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.axvline(0, color="#1a1f3a", linewidth=0.8, linestyle="--", alpha=0.5)

fig.suptitle(
    "Weight Distributions in Layer 0",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()
show(fig, filename="03-weight-distributions.png")

# %% [markdown]
# The distributions are approximately Gaussian, centered near zero. This is
# typical for trained transformer weights and is why Normal Float 4 (NF4)
# quantization works so well — it uses quantile values of a normal
# distribution as bin centers (you'll explore this in Lab 09).

# %% [markdown]
# ---
# ## 10. Try Other Layers
#
# Let's verify our block works for more than just layer 0.

# %%
from microscale.transformer_block import (
    TransformerBlock as TB,
)
from microscale.transformer_block import (
    _compute_rope_frequencies,
    load_qwen3_layer_weights,
)

layers_to_test = [0, 7, 14, 21, 27]
results = []

for layer_idx in layers_to_test:
    block = TB().to(device)
    load_qwen3_layer_weights(block, ref_model, layer_idx=layer_idx)
    block.eval()

    # Capture input AND output via hooks (not hidden_states indexing,
    # because hidden_states[-1] includes the final model.norm for the
    # last layer, which would cause a false mismatch).
    layer_in = []
    layer_ref_out = []

    def pre_hook(module, args, kwargs, _buf=layer_in):
        _buf.append(args[0].clone())

    def post_hook(module, args, kwargs, output, _buf=layer_ref_out):
        _buf.append(output[0].clone())

    h1 = ref_model.model.layers[layer_idx].register_forward_pre_hook(
        pre_hook,
        with_kwargs=True,
    )
    h2 = ref_model.model.layers[layer_idx].register_forward_hook(
        post_hook,
        with_kwargs=True,
    )
    with torch.no_grad():
        ref_model(**inputs)
    h1.remove()
    h2.remove()

    layer_input = layer_in[0]
    layer_output = layer_ref_out[0]

    layer_cos, layer_sin = _compute_rope_frequencies(
        128,
        layer_input.shape[1],
        theta=1e6,
        device=device,
    )
    with torch.no_grad():
        our_out = block(layer_input, layer_cos, layer_sin)

    diff = (our_out - layer_output).abs().max().item()
    ok = torch.allclose(our_out, layer_output, atol=1e-5)
    results.append((layer_idx, diff, ok))

table = Table(title="Verification Across Layers")
table.add_column("Layer", justify="right", style="bold")
table.add_column("Max Diff", justify="right")
table.add_column("Status")

for layer_idx, diff, ok in results:
    status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
    table.add_row(str(layer_idx), f"{diff:.2e}", status)

console.print(table)

# %% [markdown]
# All layers match — our implementation is not specific to layer 0. It is a
# correct, general-purpose Qwen3 decoder block.

# %% [markdown]
# ---
# ## What You Learned
#
# | Component | What it does | Key detail |
# |---|---|---|
# | RMSNorm | Normalize by root-mean-square | Float32 upcast for stability |
# | RoPE | Encode position via rotation | Split-half, not interleaved |
# | GQA | Multi-head attention with shared KV | 16 Q heads, 8 KV heads |
# | QK-Norm | Normalize Q and K per-head | Before RoPE, after reshape |
# | SwiGLU | Gated FFN with SiLU | gate * up, then down |
# | Decoder Layer | Pre-norm with two residuals | norm → attn → add → norm → ffn → add |
#
# ### The artifact
#
# The complete implementation lives in `microscale/transformer_block.py`.
# You can load any Qwen3 layer and produce identical outputs:
#
# ```python
# from microscale.transformer_block import (
#     TransformerBlock, load_qwen3_layer_weights, _compute_rope_frequencies
# )
# block = TransformerBlock()
# load_qwen3_layer_weights(block, hf_model, layer_idx=5)
# cos, sin = _compute_rope_frequencies(128, seq_len, theta=1e6)
# output = block(input_tensor, cos, sin)
# ```
#
# ### Artifacts in `outputs/`
#
# | File | What it shows |
# |------|---------------|
# | `03-rope-frequencies.png` | RoPE cosine and sine tables |
# | `03-weight-distributions.png` | Weight histograms for all projections |
#
# ### References
#
# - Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
# - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
# - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
# - Shazeer, "GLU Variants Improve Transformer" (2020)
