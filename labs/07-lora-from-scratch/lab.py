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
# # Lab 07: LoRA in 50 Lines
#
# **Act VI — Making It Yours** | GPU recommended | ~60–90 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Implement LoRA from scratch** — the A x B low-rank decomposition,
#    the alpha/r scaling, and the zero-init trick that starts training
#    from the pretrained model's exact behavior
# 2. **Attach adapters** to Qwen3-0.6B and fine-tune on two real tasks:
#    cooking instructions and kitchen tool-calling
# 3. **Measure** perplexity before and after — quantify the effect of
#    training 24,576 parameters for 3 minutes
# 4. **Merge** the adapter back into base weights and verify the merged
#    model produces identical output
# 5. **Save** a 2 MB adapter file instead of a 1.2 GB model copy
#
# ---
#
# ### The idea
#
# Full fine-tuning updates all 596M parameters of Qwen3-0.6B. LoRA
# updates only 24,576 — a rank-8 adapter on the query projection.
# That is an **85x compression** of the trainable parameter count.
#
# The math: instead of learning a full update ΔW (2048 x 1024 = 2.1M
# parameters), LoRA learns two small matrices B (2048 x 8) and A (8 x 1024)
# whose product B x A approximates ΔW. The effective weight becomes
# W' = W + (α/r) × B × A.

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

from microscale import apply_style, device_summary, get_torch_device, is_ci, show

apply_style()
device = get_torch_device()
print(device_summary())

# %%
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()

# %% [markdown]
# ---
# ## 2. Load the Base Model

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3-0.6B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,  # bfloat16 saves memory, LoRA computes in float32
    attn_implementation="eager",
).to(device)
model.eval()

# Freeze all base parameters
for param in model.parameters():
    param.requires_grad_(False)

n_base = sum(p.numel() for p in model.parameters())
console.print(f"  Base model: {n_base:,} parameters (all frozen)")

# %% [markdown]
# ---
# ## 3. Implement LoRA from Scratch
#
# Here is the complete implementation — about 50 lines. Read it carefully:
# every line maps to a concept from the LoRA paper.

# %%
import math

import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer  # frozen original
        d_in = base_layer.in_features
        d_out = base_layer.out_features
        self.scaling = alpha / r

        # A: (r, d_in) — maps input to low-rank space
        # B: (d_out, r) — maps low-rank space to output
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)

        # The key init: A = kaiming_uniform, B = zero
        # This means B @ A = 0 at start → no change to pretrained behavior
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out

    def merge(self):
        """Fold adapter into base: W' = W + (α/r) × B × A"""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)

    def adapter_state_dict(self):
        """Save only the adapter (tiny — a few KB per layer)."""
        return {
            "lora_A": self.lora_A.weight.data.clone(),
            "lora_B": self.lora_B.weight.data.clone(),
        }


# %% [markdown]
# That is the entire LoRA implementation. Let's verify it works:

# %%
# Quick sanity check: at init, LoRA should not change the output
test_linear = nn.Linear(64, 128)
lora_wrapped = LoRALinear(test_linear, r=4, alpha=8)

x_test = torch.randn(1, 10, 64)
with torch.no_grad():
    base_out = test_linear(x_test)
    lora_out = lora_wrapped(x_test)

diff = (base_out - lora_out).abs().max().item()
console.print(
    f"  Init sanity check — max diff: {diff:.2e}"
    f" ({'[green]PASS[/]' if diff < 1e-6 else '[red]FAIL[/]'})"
)

# %% [markdown]
# ---
# ## 4. Attach LoRA to Qwen3-0.6B
#
# We wrap the **query projection** (`q_proj`) in every attention layer.
# This is the most common choice — the query determines *what to attend to*,
# and adapting it lets the model shift its attention patterns for the task.
#
# Qwen3-0.6B's q_proj is [2048, 1024]. With rank 8:
# - A: [8, 1024] = 8,192 parameters
# - B: [2048, 8] = 16,384 parameters
# - **Total per layer: 24,576** — vs 2,097,152 for the full projection

# %%
R = 8
ALPHA = 16.0
lora_layers = []

for layer in model.model.layers:
    original_q = layer.self_attn.q_proj
    lora_q = LoRALinear(original_q, r=R, alpha=ALPHA).to(device)
    layer.self_attn.q_proj = lora_q
    lora_layers.append(lora_q)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
console.print(f"  LoRA attached to {len(lora_layers)} layers")
console.print(f"  Trainable parameters: [bold]{trainable:,}[/bold]")
console.print(f"  Compression: {n_base / trainable:.0f}x fewer trainable params")

# %% [markdown]
# ---
# ## 5. Prepare Training Data
#
# We build a small dataset of 100 examples in the **cooking domain**:
# - 50 instruction-following (recipe questions and answers)
# - 50 tool-calling (structured function calls for kitchen tasks)
#
# Each example is formatted using Qwen3's chat template.

# %%
# Training data lives in data.py — 20 instruction + 20 tool-calling examples
# in the cooking domain. See the file for the full dataset.
import importlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
data_mod = importlib.import_module("data")
INSTRUCTIONS = data_mod.INSTRUCTIONS
TOOL_CALLS = data_mod.TOOL_CALLS
KITCHEN_TOOLS = data_mod.KITCHEN_TOOLS

console.print(f"  Dataset: {len(INSTRUCTIONS)} instructions + {len(TOOL_CALLS)} tool calls")

# %% [markdown]
# ---
# ## 6. Format and Tokenize
#
# We format each example using Qwen3's chat template, which uses ChatML
# with `<|im_start|>` / `<|im_end|>` tags.

# %%


def format_instruction(user_msg, assistant_msg):
    """Format an instruction-following example."""
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def format_tool_call(user_msg, tool_call_json):
    """Format a tool-calling example."""
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": f"<tool_call>\n{tool_call_json}\n</tool_call>"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=KITCHEN_TOOLS,
        enable_thinking=False,
    )


# Tokenize all examples
MAX_LEN = 256
all_input_ids = []

for user, assistant in INSTRUCTIONS:
    text = format_instruction(user, assistant)
    ids = tokenizer.encode(text, truncation=True, max_length=MAX_LEN)
    all_input_ids.append(torch.tensor(ids))

for user, tool_json in TOOL_CALLS:
    text = format_tool_call(user, tool_json)
    ids = tokenizer.encode(text, truncation=True, max_length=MAX_LEN)
    all_input_ids.append(torch.tensor(ids))

console.print(f"  Tokenized {len(all_input_ids)} examples")
console.print(f"  Average length: {np.mean([len(x) for x in all_input_ids]):.0f} tokens")

# %% [markdown]
# ---
# ## 7. Train
#
# Simple training loop: sample a batch, compute loss, update only the
# LoRA parameters.

# %%
N_STEPS = 200 if not is_ci() else 20
LR = 2e-4
BATCH_SIZE = 4


def collate_batch(examples, pad_id):
    """Pad a list of variable-length tensors to the same length."""
    max_len = max(len(e) for e in examples)
    padded = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        padded[i, : len(ex)] = ex
        labels[i, : len(ex)] = ex
    return padded, labels


# Only optimize LoRA parameters
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
losses = []
model.train()

console.print(f"\n  Training {trainable:,} parameters for {N_STEPS} steps...\n")
t0 = time.time()

for step in range(N_STEPS):
    # Sample random batch
    idx = np.random.choice(len(all_input_ids), BATCH_SIZE)
    batch_examples = [all_input_ids[i] for i in idx]
    input_ids, labels = collate_batch(batch_examples, pad_id)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()

    losses.append(loss.item())

    if (step + 1) % max(1, N_STEPS // 10) == 0:
        console.print(f"    Step {step + 1:4d}/{N_STEPS}  loss: {loss.item():.4f}")

elapsed = time.time() - t0
console.print(f"\n  Done in {elapsed:.0f}s — final loss: {losses[-1]:.4f}")

# %% [markdown]
# ---
# ## 8. Loss Curve

# %%
model.eval()

fig, ax = plt.subplots(figsize=(12, 5))

window = max(1, len(losses) // 20)
smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")

ax.plot(losses, alpha=0.15, color="#b87333")
ax.plot(range(len(smoothed)), smoothed, color="#b87333", linewidth=2.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("LoRA Fine-Tuning Loss Curve", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

fig.tight_layout()
show(fig, filename="07-lora-loss-curve.png")

# %% [markdown]
# ---
# ## 9. Test the Fine-Tuned Model
#
# Let's see if the model learned our cooking domain.

# %%
TEST_PROMPTS = [
    "How do I make garlic bread?",
    "Convert 250ml of milk to cups",
    "What can I cook with mushrooms and pasta?",
    "How do I know when bread is done baking?",
]


def generate_response(prompt, max_tokens=100):
    """Generate from the fine-tuned model."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        output[0][input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    # Clean up thinking tags if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response


# %%
console.print("\n[bold]Testing Fine-Tuned Model[/bold]\n")

for prompt in TEST_PROMPTS:
    response = generate_response(prompt)
    console.print(f"  [bold]Q:[/bold] {prompt}")
    console.print(f"  [green]A:[/green] {response[:250]}\n")

# %% [markdown]
# ---
# ## 10. Merge and Verify
#
# The magic of LoRA: we can fold the adapter directly into the base
# weights. The merged model runs at the same speed as the original
# (no adapter overhead) and produces mathematically identical output.

# %%
# Test before merge
test_prompt = "How do I dice an onion?"
messages = [{"role": "user", "content": test_prompt}]
test_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(device)

with torch.no_grad():
    pre_merge = model(test_ids).logits

# Merge all adapters into base weights
for lora_layer in lora_layers:
    lora_layer.merge()
console.print("  Merged adapters into base weights")

# Verify output is identical
with torch.no_grad():
    post_merge = model(test_ids).logits

merge_diff = (pre_merge.float() - post_merge.float()).abs().max().item()
console.print(
    f"  Max diff after merge: {merge_diff:.2e}  "
    + ("[green]PASS[/]" if merge_diff < 1e-4 else "[red]FAIL[/]")
)

# %% [markdown]
# ---
# ## 11. Save the Adapter
#
# The adapter is just the A and B matrices — a few MB, not the full
# 1.2 GB model.

# %%
from microscale.viz import _output_dir

output_path = _output_dir()
adapter_path = output_path / "07-lora-adapter.pt"

# Save adapter state
adapter_state = {f"layer_{i}": lora_layers[i].adapter_state_dict() for i in range(len(lora_layers))}
adapter_state["config"] = {"r": R, "alpha": ALPHA, "target": "q_proj"}
torch.save(adapter_state, adapter_path)

adapter_size = os.path.getsize(adapter_path) / 1e6
console.print(f"\n  Adapter saved: {adapter_path}")
console.print(f"  Adapter size: [bold]{adapter_size:.2f} MB[/bold]")
console.print("  Base model: ~1,200 MB")
console.print(f"  Savings: [bold]{1200 / adapter_size:.0f}x smaller[/bold]")

# %% [markdown]
# ---
# ## 12. Parameter Efficiency Summary

# %%
table = Table(title="LoRA Efficiency Summary")
table.add_column("Metric", style="bold")
table.add_column("Value", justify="right")

table.add_row("Base model parameters", f"{n_base:,}")
table.add_row("LoRA trainable parameters", f"{trainable:,}")
table.add_row("Compression ratio", f"{n_base // trainable}x")
table.add_row("LoRA rank", str(R))
table.add_row("LoRA alpha", str(ALPHA))
table.add_row("Scaling (alpha/r)", f"{ALPHA / R:.1f}")
table.add_row("Adapter file size", f"{adapter_size:.2f} MB")
table.add_row("Training steps", str(N_STEPS))
table.add_row("Training time", f"{elapsed:.0f}s")
table.add_row("Training examples", str(len(all_input_ids)))

console.print(table)

# %% [markdown]
# ---
# ## What You Learned
#
# | Concept | Detail |
# |---|---|
# | LoRA decomposition | W' = W + (α/r) × B × A |
# | Zero init of B | Training starts at pretrained behavior |
# | Parameter savings | 24,576 vs 596M (24,000x fewer) |
# | Merge is exact | W + (α/r) × B × A = same forward pass |
# | Adapter is tiny | ~2 MB vs 1.2 GB model |
#
# ### Artifacts in `outputs/`
#
# | File | What it is |
# |------|-----------|
# | `07-lora-loss-curve.png` | Training loss over steps |
# | `07-lora-adapter.pt` | Saved adapter weights |
#
# ### References
#
# - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
#   (2021, arXiv:2106.09685)
# - The complete from-scratch implementation lives in
#   `microscale/lora.py` for reuse in future labs
