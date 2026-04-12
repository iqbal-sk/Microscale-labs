# microscale/transformer_block.py
"""A transformer decoder block built from raw PyTorch operations.

Every component matches HuggingFace's Qwen3 implementation exactly.
Load real weights from any Qwen3 layer and verify with torch.allclose.

Components:
    - RMSNorm: Root Mean Square Layer Normalization
    - RotaryEmbedding: Rotary Position Embeddings (RoPE)
    - GroupedQueryAttention: GQA with QK-Norm
    - SwiGLUFFN: Gated Linear Unit with SiLU activation
    - TransformerBlock: Full decoder layer assembling all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center the activations (no mean
    subtraction). It only normalizes by the root-mean-square, then scales
    by a learned weight vector.

    Formula: output = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability, then cast back
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


def _compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for rotary embeddings.

    Args:
        head_dim: Dimension per attention head (128 for Qwen3-0.6B).
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency (1,000,000 for Qwen3-0.6B).
        device: Target device.

    Returns:
        (cos, sin): Each of shape (max_seq_len, head_dim).
    """
    # Inverse frequencies: theta^(-2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )  # shape: (head_dim // 2,)

    # Position indices
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Outer product: (max_seq_len,) x (head_dim // 2,) -> (max_seq_len, head_dim // 2)
    freqs = torch.outer(positions, inv_freq)

    # Double up: cat(freqs, freqs) -> (max_seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension in half, negate the second half, and swap.

    [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]

    This is the Qwen3/Llama-style rotation (split-half, NOT interleaved).
    """
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
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor, shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor, shape (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine table, shape (seq_len, head_dim).
        sin: Sine table, shape (seq_len, head_dim).

    Returns:
        Rotated (q, k) with same shapes.
    """
    # Reshape for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Grouped Query Attention with QK-Norm
# ---------------------------------------------------------------------------


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads.

    Args:
        x: shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head.

    Returns:
        shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with QK-Norm and RoPE.

    Qwen3-0.6B specifics:
        - 16 query heads, 8 KV heads (group ratio = 2)
        - head_dim = 128 (so Q projection is [2048, 1024] — upsamples!)
        - QK-Norm: RMSNorm applied per-head AFTER reshape, BEFORE RoPE
        - No bias on any projection
    """

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

        # QK-Norm: per-head RMSNorm on the head_dim dimension
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            cos: (seq_len, head_dim) — precomputed RoPE cosines
            sin: (seq_len, head_dim) — precomputed RoPE sines

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = hidden_states.shape

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)  # (B, S, num_heads * head_dim)
        k = self.k_proj(hidden_states)  # (B, S, num_kv_heads * head_dim)
        v = self.v_proj(hidden_states)  # (B, S, num_kv_heads * head_dim)

        # 2. Reshape to heads: (B, S, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 3. QK-Norm (per-head, BEFORE transpose, BEFORE RoPE)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Transpose to (B, num_heads, S, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 6. Repeat KV heads to match Q heads (GQA)
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        # 7. Attention scores: Q @ K^T / sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # 8. Causal mask (additive): prevent attending to future positions
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

        # 9. Softmax in float32 for stability, then cast back
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)

        # 10. Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, D)

        # 11. Reshape back: (B, H, S, D) -> (B, S, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch, seq_len, -1)

        # 12. Output projection
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    The "gate" and "up" projections expand from hidden_size to intermediate_size.
    The SiLU activation (also called "swish") is applied to the gate path.
    Element-wise multiplication between gate and up paths is the "GLU" operation.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full Transformer Decoder Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """A single transformer decoder layer matching Qwen3's architecture.

    Forward pass:
        residual = x
        x = input_layernorm(x)
        x = attention(x, cos, sin)
        x = residual + x          # first residual connection
        residual = x
        x = post_attention_layernorm(x)
        x = ffn(x)
        x = residual + x          # second residual connection
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 3072,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            cos, sin: RoPE tables, each (seq_len, head_dim)

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Attention block with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        # FFN block with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_qwen3_layer_weights(
    block: TransformerBlock,
    hf_model,
    layer_idx: int = 0,
) -> None:
    """Copy weights from a HuggingFace Qwen3 model layer into our block.

    Args:
        block: Our TransformerBlock instance.
        hf_model: A loaded Qwen3ForCausalLM model.
        layer_idx: Which layer to copy from (0-indexed).
    """
    hf_layer = hf_model.model.layers[layer_idx]

    # Attention projections
    block.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
    block.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
    block.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
    block.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)

    # QK-Norm weights
    block.self_attn.q_norm.weight.data.copy_(hf_layer.self_attn.q_norm.weight.data)
    block.self_attn.k_norm.weight.data.copy_(hf_layer.self_attn.k_norm.weight.data)

    # Layer norms
    block.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
    block.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)

    # FFN projections
    block.mlp.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
    block.mlp.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
    block.mlp.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)
