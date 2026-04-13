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
# # Lab 07-B: LoRA for Tool Calling
#
# **Act VI — Making It Yours** | GPU recommended | ~60–90 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Reuse** the LoRA module from Lab 07-A via `microscale.lora` —
#    see how the same building block solves a different problem
# 2. **Attach** higher-rank adapters (r=16) to **both** `q_proj` and
#    `v_proj` — tool calling needs more capacity than behavioral tuning
# 3. **Train** on 20 kitchen tool-calling examples using Qwen3's native
#    tool-calling chat template
# 4. **Evaluate** with a deterministic parser: extract JSON, validate
#    function names, check arguments against the tool schema
# 5. **Test generalization** on held-out prompts not seen in training
#
# ---
#
# ### The idea
#
# Lab 07-A taught the model cooking *knowledge* through natural-language
# pairs. This lab teaches the model cooking *actions* — emitting
# structured JSON that invokes functions like `get_recipe()` or
# `set_timer()`.
#
# Structured output is harder than free text. The model must:
# - Pick the right function from 6 options
# - Generate valid JSON syntax
# - Match the schema exactly (field names, types)
# - Wrap output in `<tool_call>...</tool_call>` tags
#
# For this, we use:
# - **Rank 16** (vs 8 in Lab 07-A): more capacity for structured format
# - **Target q_proj + v_proj** (vs just q_proj): more places to adapt
# - **Qwen3's tool-calling template** with `<tools>` in the system prompt

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

from microscale import apply_style, device_summary, get_torch_device, is_ci, show

apply_style()
device = get_torch_device()
print(device_summary())

# %%
import importlib
import json
import os
import re
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
    dtype=torch.bfloat16,
    attn_implementation="eager",
).to(device)
model.eval()

# Freeze base parameters
for param in model.parameters():
    param.requires_grad_(False)

n_base = sum(p.numel() for p in model.parameters())
console.print(f"  Base model: {n_base:,} parameters (frozen)")

# %% [markdown]
# ---
# ## 3. Load the Dataset
#
# Our dataset has 20 training examples + 6 held-out test prompts. The
# tools are defined as a JSON schema list that gets inserted into the
# system prompt via Qwen3's chat template.

# %%
from microscale.datasets.cooking_tools import (
    KITCHEN_TOOLS,
    TEST_PROMPTS,
    TOOL_CALLS,
)

console.print(f"  Training examples: {len(TOOL_CALLS)}")
console.print(f"  Held-out test prompts: {len(TEST_PROMPTS)}")
console.print(f"  Available tools: {len(KITCHEN_TOOLS)}")

# Show the 6 functions
table = Table(title="Kitchen Assistant API")
table.add_column("Function", style="bold")
table.add_column("Description")
table.add_column("Required Args")
for tool in KITCHEN_TOOLS:
    fn = tool["function"]
    required = ", ".join(fn["parameters"].get("required", []))
    table.add_row(fn["name"], fn["description"], required)
console.print(table)

# %% [markdown]
# ---
# ## 4. Baseline — Can the Base Model Already Do This?
#
# Before fine-tuning, let's see how Qwen3-0.6B handles tool calling out
# of the box.

# %%


def generate_tool_call(mdl, prompt, max_tokens=150):
    """Generate a response with tools in the system prompt."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tools=KITCHEN_TOOLS,
        enable_thinking=False,
    ).to(device)
    with torch.no_grad():
        output = mdl.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(
        output[0][input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def parse_tool_call(response: str) -> dict | None:
    """Extract JSON tool call from response. Returns None if parsing fails."""
    # Look for <tool_call>...</tool_call> pattern
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Also try raw JSON (some models don't wrap in tags)
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return None


def score_tool_call(response: str, tool_names: list[str]) -> dict:
    """Score a tool call: parses correctly? valid function? has required args?"""
    parsed = parse_tool_call(response)
    result = {
        "parsed": parsed is not None,
        "valid_function": False,
        "has_arguments": False,
        "parsed_call": parsed,
    }
    if parsed:
        result["valid_function"] = parsed.get("name") in tool_names
        result["has_arguments"] = "arguments" in parsed and isinstance(
            parsed.get("arguments"),
            dict,
        )
    return result


tool_names = [t["function"]["name"] for t in KITCHEN_TOOLS]

console.print("\n[bold]Baseline (before fine-tuning)[/bold]\n")
baseline_scores = []
baseline_responses = {}
for prompt in TEST_PROMPTS:
    resp = generate_tool_call(model, prompt)
    baseline_responses[prompt] = resp
    score = score_tool_call(resp, tool_names)
    baseline_scores.append(score)

    status = "[green]OK[/]" if score["parsed"] and score["valid_function"] else "[red]FAIL[/]"
    fn = score["parsed_call"]["name"] if score["parsed"] else "-"
    console.print(f"  {status}  [bold]{prompt[:45]:45s}[/bold] -> fn={fn}")

# %% [markdown]
# ---
# ## 5. Attach LoRA
#
# We use `microscale.lora` (the reusable module built in Lab 07-A) to
# attach LoRA to both the query AND value projections. Tool calling
# needs more places for the model to adapt.

# %%
from microscale.lora import apply_lora

lora_layers = apply_lora(
    model,
    target_modules=["q_proj", "v_proj"],  # Both Q and V
    r=16,
    alpha=32.0,
)

# Move LoRA params to device
for layer in lora_layers:
    layer.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
console.print(f"\n  Attached LoRA to {len(lora_layers)} layers (q_proj + v_proj)")
console.print(f"  Trainable parameters: [bold]{trainable:,}[/bold]")
console.print(f"  Compression: {n_base / trainable:.0f}x")

# %% [markdown]
# ---
# ## 6. Format and Tokenize
#
# Tool-calling examples use Qwen3's chat template with `tools=` argument,
# which embeds the tool schemas in a `<tools>` block inside the system
# prompt automatically.

# %%


def format_tool_example(user_msg, tool_call_json):
    """Format a tool-calling training example."""
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


MAX_LEN = 512
all_input_ids = []
for user, tool_json in TOOL_CALLS:
    text = format_tool_example(user, tool_json)
    ids = tokenizer.encode(text, truncation=True, max_length=MAX_LEN)
    all_input_ids.append(torch.tensor(ids))

lengths = [len(x) for x in all_input_ids]
console.print(
    f"  Tokenized {len(all_input_ids)} examples  (avg length: {np.mean(lengths):.0f} tokens)"
)

# %% [markdown]
# ---
# ## 7. Train
#
# Standard LoRA training loop. Note the higher learning rate than DPO —
# we want to move the model meaningfully, not fine-tune preferences.

# %%
N_STEPS = 300 if not is_ci() else 30
LR = 2e-4
BATCH_SIZE = 4


def collate_batch(examples, pad_id):
    max_len = max(len(e) for e in examples)
    padded = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        padded[i, : len(ex)] = ex
        labels[i, : len(ex)] = ex
    return padded, labels


lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

losses = []
model.train()
t0 = time.time()

log_interval = max(1, N_STEPS // 10)
console.print(f"\n  Training {trainable:,} parameters for {N_STEPS} steps...\n")

for step in range(N_STEPS):
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

    if (step + 1) % log_interval == 0:
        avg = np.mean(losses[-log_interval:])
        console.print(f"  Step {step + 1:4d}/{N_STEPS}  loss={avg:.4f}  [{time.time() - t0:.0f}s]")

console.print(
    f"\n  Training done in {time.time() - t0:.0f}s  final loss: {np.mean(losses[-20:]):.4f}"
)

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
ax.set_title(
    "Tool-Calling LoRA Training Loss",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)
fig.tight_layout()
show(fig, filename="07-b-loss-curve.png")

# %% [markdown]
# ---
# ## 9. Evaluate — Did the Model Learn to Call Tools?

# %%
console.print("\n[bold]After fine-tuning — held-out prompts[/bold]\n")

after_scores = []
after_responses = {}
for prompt in TEST_PROMPTS:
    resp = generate_tool_call(model, prompt)
    after_responses[prompt] = resp
    score = score_tool_call(resp, tool_names)
    after_scores.append(score)

# Results table
table = Table(title="Tool-Calling Evaluation")
table.add_column("Prompt", max_width=40)
table.add_column("Baseline", justify="center")
table.add_column("After LoRA", justify="center")
table.add_column("Function Called", justify="center")

for prompt, base, after in zip(TEST_PROMPTS, baseline_scores, after_scores):
    base_ok = "[green]OK[/]" if (base["parsed"] and base["valid_function"]) else "[red]x[/]"
    after_ok = "[green]OK[/]" if (after["parsed"] and after["valid_function"]) else "[red]x[/]"
    fn = after["parsed_call"]["name"] if after["parsed"] else "-"
    table.add_row(prompt[:40], base_ok, after_ok, fn)

console.print(table)

# %% [markdown]
# ---
# ## 10. Quantify the Improvement


# %%
def summarize(scores):
    n = len(scores)
    return {
        "parsed": sum(s["parsed"] for s in scores),
        "valid_fn": sum(s["parsed"] and s["valid_function"] for s in scores),
        "has_args": sum(s["parsed"] and s["has_arguments"] for s in scores),
        "total": n,
    }


base_sum = summarize(baseline_scores)
after_sum = summarize(after_scores)

fig, ax = plt.subplots(figsize=(12, 6))
categories = ["Valid JSON\nparsed", "Valid function\nname", "Has arguments\nfield"]
base_vals = [base_sum["parsed"], base_sum["valid_fn"], base_sum["has_args"]]
after_vals = [after_sum["parsed"], after_sum["valid_fn"], after_sum["has_args"]]

x = np.arange(len(categories))
width = 0.35
b1 = ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#8b3a3a", edgecolor="#1a1f3a")
b2 = ax.bar(
    x + width / 2, after_vals, width, label="After LoRA", color="#4a7c74", edgecolor="#1a1f3a"
)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.1,
            f"{int(h)}/{base_sum['total']}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Count (out of held-out test prompts)")
ax.set_title(
    "Tool-Calling Success: Baseline vs After LoRA",
    fontsize=13,
    fontweight="bold",
)
ax.legend()
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, base_sum["total"] + 1)
fig.tight_layout()
show(fig, filename="07-b-evaluation.png")

console.print(
    f"\n  Baseline valid calls: [bold]{base_sum['valid_fn']}/{base_sum['total']}[/bold]"
    f" ({base_sum['valid_fn'] / base_sum['total']:.0%})"
)
console.print(
    f"  After LoRA valid calls: [bold]{after_sum['valid_fn']}/{after_sum['total']}[/bold]"
    f" ({after_sum['valid_fn'] / after_sum['total']:.0%})"
)

# %% [markdown]
# ---
# ## 11. Show a Full Example
#
# Let's look at one complete response to see what the model now produces.

# %%
example_prompt = TEST_PROMPTS[0]
console.print(f"\n  [bold]Prompt:[/bold] {example_prompt}\n")
console.print("  [dim]Baseline response:[/dim]")
console.print(f"    {baseline_responses[example_prompt][:300]}\n")
console.print("  [green]After LoRA response:[/green]")
console.print(f"    {after_responses[example_prompt][:300]}\n")

# %% [markdown]
# ---
# ## 12. Save the Adapter

# %%
from microscale.viz import _output_dir

output_path = _output_dir()
adapter_path = output_path / "07-b-tool-calling-adapter.pt"

adapter_state = {f"layer_{i}": lora_layers[i].adapter_state_dict() for i in range(len(lora_layers))}
adapter_state["config"] = {
    "r": 16,
    "alpha": 32.0,
    "target": ["q_proj", "v_proj"],
}
torch.save(adapter_state, adapter_path)

size_mb = os.path.getsize(adapter_path) / 1e6
console.print(f"\n  Adapter saved: {adapter_path}")
console.print(f"  Size: [bold]{size_mb:.2f} MB[/bold]")

# %% [markdown]
# ---
# ## What You Learned
#
# | Concept | Detail |
# |---|---|
# | Reusing LoRA module | `microscale.lora` works for any task |
# | Higher rank for structure | r=16 beats r=8 for JSON generation |
# | Multiple target modules | q_proj + v_proj > q_proj alone |
# | Qwen3 tool template | Automatic `<tools>` schema insertion |
# | Deterministic evaluation | Parse JSON, verify function name/args |
# | Generalization | Held-out prompts not seen during training |
#
# ### Artifacts in `outputs/`
#
# | File | What it is |
# |------|-----------|
# | `07-b-loss-curve.png` | Training loss over steps |
# | `07-b-evaluation.png` | Baseline vs after comparison |
# | `07-b-tool-calling-adapter.pt` | Saved adapter |
#
# ### Compare with Lab 07-A
#
# | | Lab 07-A (Behavioral) | Lab 07-B (Tool Calling) |
# |---|---|---|
# | Task | Free-text response style | Structured JSON output |
# | Rank | 8 | 16 |
# | Target modules | q_proj | q_proj + v_proj |
# | Evaluation | Side-by-side reading | Deterministic JSON parsing |
#
# ### References
#
# - Hu et al., "LoRA" (2021, arXiv:2106.09685)
# - Qwen3 tool-calling template documentation
