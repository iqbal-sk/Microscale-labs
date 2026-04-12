# tests/test_metrics.py
import numpy as np

from microscale.metrics import compute_per_token_loss, compute_perplexity


def test_compute_perplexity_returns_float(monkeypatch):
    """Test that compute_perplexity returns a positive float.

    Uses a mock model to avoid downloading real weights in CI.
    """
    import torch

    class MockOutput:
        def __init__(self, loss):
            self.loss = loss

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def forward(self, input_ids=None, labels=None, **kwargs):
            # Return a fixed loss of 2.0
            return MockOutput(loss=torch.tensor(2.0))

    class MockTokenizer:
        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    model = MockModel()
    tokenizer = MockTokenizer()
    ppl = compute_perplexity(model, tokenizer, ["Hello world"], device=torch.device("cpu"))
    assert isinstance(ppl, float)
    assert ppl > 0
    # exp(2.0) ≈ 7.389
    assert abs(ppl - np.exp(2.0)) < 0.01


def test_compute_per_token_loss_returns_correct_lengths(monkeypatch):
    """Test shape consistency of per-token loss."""
    import torch

    class MockOutput:
        def __init__(self, seq_len):
            self.logits = torch.randn(1, seq_len, 100)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            return self

        def forward(self, input_ids=None, **kwargs):
            return MockOutput(seq_len=input_ids.shape[1])

    class MockTokenizer:
        def __call__(self, text, return_tensors=None, **kwargs):
            return {"input_ids": torch.tensor([[10, 20, 30, 40, 50]])}

        def decode(self, token_id):
            return f"tok{token_id}"

    model = MockModel()
    tokenizer = MockTokenizer()
    tokens, losses = compute_per_token_loss(model, tokenizer, "test", device=torch.device("cpu"))
    assert len(tokens) == 5
    assert len(losses) == 5
    assert np.isnan(losses[0])  # First token has no loss
    assert all(isinstance(val, float) for val in losses)
