# microscale/metrics.py
"""Evaluation metrics for language models — perplexity, loss, scoring."""

import numpy as np
import torch
from tqdm import tqdm


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device | None = None,
    max_length: int = 512,
) -> float:
    """Compute perplexity over a list of texts.

    Correctly averages log-likelihoods across all tokens, then exponentiates.
    This avoids the common mistake of averaging per-sentence perplexities.

    Args:
        model: A HuggingFace CausalLM model.
        tokenizer: The matching tokenizer.
        texts: List of strings to evaluate.
        device: Torch device. If None, uses model's device.
        max_length: Maximum token length per text (truncates if longer).

    Returns:
        Perplexity (float). Lower is better.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity", leave=False):
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encodings["input_ids"].to(device)
            # Labels = input_ids; HuggingFace internally shifts left by 1
            outputs = model(input_ids=input_ids, labels=input_ids)
            # outputs.loss is mean cross-entropy over valid tokens
            seq_len = input_ids.shape[1]
            # Number of predicted tokens = seq_len - 1 (first token has no target)
            n_tokens = seq_len - 1
            if n_tokens > 0:
                total_nll += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(np.exp(avg_nll))


def compute_per_token_loss(
    model,
    tokenizer,
    text: str,
    device: torch.device | None = None,
) -> tuple[list[str], list[float]]:
    """Compute per-token cross-entropy loss for a single text.

    Returns:
        (tokens, losses): Lists of token strings and their corresponding losses.
        The first token has no loss (nothing to predict from).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[:-1]  # (seq_len-1, vocab_size)
    shift_labels = input_ids[0, 1:]  # (seq_len-1,)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_losses = loss_fn(shift_logits, shift_labels).cpu().tolist()

    tokens = [tokenizer.decode(t) for t in input_ids[0].tolist()]
    # First token has no loss; losses align with tokens[1:]
    return tokens, [float("nan")] + per_token_losses
