# tests/test_transformer_block.py
"""Tests for the from-scratch transformer block.

Unit tests use synthetic data. The integration test (marked slow) loads
the real Qwen3-0.6B model and verifies torch.allclose.
"""

import torch

from microscale.transformer_block import (
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
    _compute_rope_frequencies,
    _repeat_kv,
    _rotate_half,
    apply_rotary_pos_emb,
)


def test_rmsnorm_output_shape():
    norm = RMSNorm(64, eps=1e-6)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_rmsnorm_unit_input():
    """RMSNorm of all-ones should return the weight vector."""
    norm = RMSNorm(4, eps=1e-6)
    norm.weight.data.fill_(2.0)
    x = torch.ones(1, 1, 4)
    out = norm(x)
    # RMS of all-ones = 1.0, so output = weight * 1.0 = weight
    assert torch.allclose(out[0, 0], torch.tensor([2.0, 2.0, 2.0, 2.0]), atol=1e-5)


def test_rotate_half():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = _rotate_half(x)
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    assert torch.allclose(rotated, expected)


def test_rope_frequencies_shape():
    cos, sin = _compute_rope_frequencies(128, max_seq_len=32, theta=1e6)
    assert cos.shape == (32, 128)
    assert sin.shape == (32, 128)


def test_rope_position_zero_is_identity_for_cos():
    """At position 0, cos should be 1 and sin should be 0 (no rotation)."""
    cos, sin = _compute_rope_frequencies(128, max_seq_len=1, theta=1e6)
    assert torch.allclose(cos[0], torch.ones(128), atol=1e-5)
    assert torch.allclose(sin[0], torch.zeros(128), atol=1e-5)


def test_apply_rotary_preserves_norm():
    """RoPE is a rotation — it should preserve vector norms."""
    cos, sin = _compute_rope_frequencies(64, max_seq_len=8, theta=1e6)
    q = torch.randn(1, 4, 8, 64)  # (B, H, S, D)
    k = torch.randn(1, 2, 8, 64)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    # Norms should be preserved (RoPE is a rotation)
    q_norms = q.norm(dim=-1)
    q_rot_norms = q_rot.norm(dim=-1)
    assert torch.allclose(q_norms, q_rot_norms, atol=1e-4)


def test_repeat_kv():
    x = torch.randn(1, 4, 8, 64)  # 4 KV heads
    expanded = _repeat_kv(x, n_rep=2)  # -> 8 heads
    assert expanded.shape == (1, 8, 8, 64)
    # Head 0 and 1 should be identical (repeated from KV head 0)
    assert torch.allclose(expanded[:, 0], expanded[:, 1])
    assert torch.allclose(expanded[:, 2], expanded[:, 3])


def test_swiglu_output_shape():
    ffn = SwiGLUFFN(hidden_size=64, intermediate_size=128)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64)


def test_transformer_block_output_shape():
    block = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
    )
    cos, sin = _compute_rope_frequencies(16, max_seq_len=8)
    x = torch.randn(1, 8, 64)
    out = block(x, cos, sin)
    assert out.shape == (1, 8, 64)


def test_transformer_block_residual_connection():
    """With all weights zeroed, output should equal input (residual only)."""
    block = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
    )
    # Zero all linear weights so attention and FFN produce zeros
    for param in block.parameters():
        if param.dim() >= 2:
            param.data.zero_()
    cos, sin = _compute_rope_frequencies(16, max_seq_len=8)
    x = torch.randn(1, 8, 64)
    out = block(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-5)
