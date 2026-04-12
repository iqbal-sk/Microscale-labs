# microscale/lora.py
"""LoRA (Low-Rank Adaptation) — implemented from scratch in ~50 lines.

Based on Hu et al. 2021: "LoRA: Low-Rank Adaptation of Large Language Models"

The core idea: instead of updating a full weight matrix W (d_out x d_in),
add a low-rank decomposition B @ A where A is (r x d_in) and B is (d_out x r).
This reduces trainable parameters from d_out*d_in to r*(d_in + d_out).

Key details:
    - A is initialized with kaiming_uniform (standard nn.Linear init)
    - B is initialized to zero → B @ A = 0 at start → model starts at pretrained behavior
    - Scaling factor: alpha/r (not alpha/sqrt(r), which is rsLoRA)
    - Merge: W' = W + (alpha/r) * B @ A — mathematically identical forward pass
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen nn.Linear layer.

    Usage:
        original = model.layer.q_proj          # frozen nn.Linear
        model.layer.q_proj = LoRALinear(original, r=8, alpha=16)
        # Now only LoRA params are trainable
    """

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        d_in = base_layer.in_features
        d_out = base_layer.out_features
        self.r = r
        self.scaling = alpha / r

        # Low-rank decomposition: delta_W = B @ A
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)

        # Init: A = kaiming_uniform, B = zero → starts at exact pretrained behavior
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output + scaled low-rank adapter
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merge(self) -> None:
        """Fold adapter into base weights: W' = W + (alpha/r) * B @ A."""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)

    def unmerge(self) -> None:
        """Reverse a previous merge."""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data -= delta.to(self.base_layer.weight.dtype)

    @property
    def trainable_params(self) -> int:
        """Number of trainable (adapter) parameters."""
        return self.lora_A.weight.numel() + self.lora_B.weight.numel()

    def adapter_state_dict(self) -> dict:
        """Return only the adapter parameters for saving."""
        return {
            "lora_A.weight": self.lora_A.weight.data.clone(),
            "lora_B.weight": self.lora_B.weight.data.clone(),
            "r": self.r,
            "scaling": self.scaling,
        }

    def load_adapter(self, state_dict: dict) -> None:
        """Load adapter weights from a saved state dict."""
        self.lora_A.weight.data.copy_(state_dict["lora_A.weight"])
        self.lora_B.weight.data.copy_(state_dict["lora_B.weight"])


def apply_lora(
    model,
    target_modules: list[str],
    r: int = 8,
    alpha: float = 16.0,
) -> list[LoRALinear]:
    """Apply LoRA to specified modules in a HuggingFace model.

    Args:
        model: A HuggingFace model.
        target_modules: List of module name suffixes to wrap (e.g., ["q_proj", "v_proj"]).
        r: LoRA rank.
        alpha: LoRA alpha (scaling = alpha/r).

    Returns:
        List of created LoRALinear modules.
    """
    lora_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue

        # Navigate to the parent module and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_layer = LoRALinear(module, r=r, alpha=alpha)
        setattr(parent, parts[-1], lora_layer)
        lora_modules.append(lora_layer)

    return lora_modules


def count_trainable(model) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable, total) parameter counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
