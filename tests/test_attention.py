# tests/test_attention.py
import numpy as np

from microscale.attention import (
    classify_head,
    compute_head_summary,
    diagonal_strength,
    head_entropy,
    prev_token_strength,
    sink_strength,
)


def _make_uniform(n: int) -> np.ndarray:
    """Uniform attention: every position attends equally."""
    return np.ones((n, n)) / n


def _make_sink(n: int) -> np.ndarray:
    """Sink head: all queries attend heavily to position 0."""
    mat = np.zeros((n, n))
    mat[:, 0] = 0.8
    # Distribute remaining mass uniformly
    for i in range(n):
        remaining = 1.0 - mat[i, 0]
        for j in range(1, n):
            mat[i, j] = remaining / (n - 1)
    return mat


def _make_prev_token(n: int) -> np.ndarray:
    """Previous-token head: position i attends to i-1."""
    mat = np.zeros((n, n))
    mat[0, 0] = 1.0  # First token attends to itself
    for i in range(1, n):
        mat[i, i - 1] = 0.8
        remaining = 0.2 / (n - 1)
        for j in range(n):
            if j != i - 1:
                mat[i, j] = remaining
    return mat


def test_head_entropy_uniform():
    """Uniform attention should have maximum entropy."""
    mat = _make_uniform(8)
    ent = head_entropy(mat)
    max_ent = np.log2(8)
    assert abs(ent - max_ent) < 0.01


def test_head_entropy_concentrated():
    """One-hot attention should have near-zero entropy."""
    mat = np.eye(8)  # Each position attends to itself
    ent = head_entropy(mat)
    assert ent < 0.01


def test_sink_strength_high():
    mat = _make_sink(8)
    assert sink_strength(mat) > 0.5


def test_sink_strength_low_for_uniform():
    mat = _make_uniform(8)
    assert sink_strength(mat) < 0.2


def test_prev_token_strength_high():
    mat = _make_prev_token(8)
    assert prev_token_strength(mat) > 0.5


def test_prev_token_strength_low_for_uniform():
    mat = _make_uniform(8)
    assert prev_token_strength(mat) < 0.2


def test_diagonal_strength():
    mat = np.eye(8)
    assert diagonal_strength(mat) > 0.9


def test_classify_head_sink():
    mat = _make_sink(8)
    assert classify_head(mat) == "sink"


def test_classify_head_prev_token():
    mat = _make_prev_token(8)
    assert classify_head(mat) == "previous_token"


def test_classify_head_self():
    mat = np.eye(8) * 0.7 + np.ones((8, 8)) * 0.3 / 8
    # Normalize rows
    mat = mat / mat.sum(axis=1, keepdims=True)
    assert classify_head(mat) == "self"


def test_compute_head_summary():
    # 2 layers, 3 heads, 4 tokens
    weights = [np.random.dirichlet(np.ones(4), size=(3, 4)) for _ in range(2)]
    summary = compute_head_summary(weights)
    assert summary["entropy"].shape == (2, 3)
    assert summary["sink"].shape == (2, 3)
    assert summary["prev_token"].shape == (2, 3)
    assert summary["diagonal"].shape == (2, 3)
    assert summary["classification"].shape == (2, 3)
