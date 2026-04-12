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
# # Lab 08: Your First DPO Alignment
#
# **Act VI — Making It Yours** | GPU recommended | ~90 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Understand DPO** — Direct Preference Optimization trains a model
#    to prefer one response over another, without needing a reward model
# 2. **Build preference data** — craft chosen/rejected pairs where the
#    "chosen" response is specific, safe, and helpful
# 3. **Train with TRL** — use HuggingFace's DPOTrainer with LoRA for
#    memory-efficient alignment
# 4. **Measure the shift** — compare model responses before and after
#    alignment on held-out prompts
# 5. **See the math** — the DPO loss directly optimizes for preference
#    without a separate reward model
#
# ---
#
# ### The idea
#
# Lab 07 taught the model *what* to say (cooking knowledge via SFT).
# This lab teaches it *how* to say it — preferring detailed, safe,
# specific answers over vague or dismissive ones.
#
# The DPO loss function:
#
# $$\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta \big(
# \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} -
# \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
# \big)\Big)$$
#
# In plain English: increase the probability of the chosen response
# relative to the reference model, while decreasing the probability
# of the rejected response, by exactly the amount that makes the
# implicit reward model consistent with the preference data.

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

from microscale import (
    apply_style,
    device_summary,
    get_torch_device,
    is_ci,
    show,
)

apply_style()
device = get_torch_device()
print(device_summary())

# %%
import importlib
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
    dtype=torch.bfloat16,
    attn_implementation="eager",
).to(device)

n_params = sum(p.numel() for p in model.parameters())
console.print(f"  Loaded: {n_params / 1e6:.0f}M parameters on {device}")

# %% [markdown]
# ---
# ## 3. Baseline — What Does the Unaligned Model Say?
#
# Before any training, let's see how the base model responds to
# cooking safety questions. This is our "before" snapshot.

# %%
TEST_PROMPTS = [
    "How should I handle raw chicken?",
    "Can I leave cooked rice out overnight?",
    "How do I make a good steak?",
    "My cookies are always flat. What am I doing wrong?",
    "What's the best oil for deep frying?",
]


def generate(mdl, prompt, max_tokens=150):
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)
    with torch.no_grad():
        output = mdl.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


# %%
console.print("\n[bold]Baseline Responses (before DPO)[/bold]\n")
baseline_responses = {}

for prompt in TEST_PROMPTS:
    response = generate(model, prompt)
    baseline_responses[prompt] = response
    console.print(f"  [bold]Q:[/bold] {prompt}")
    console.print(f"  [dim]A:[/dim] {response[:300]}\n")

# %% [markdown]
# ---
# ## 4. Load the Preference Data
#
# Each example has three parts:
# - **Prompt**: A cooking question
# - **Chosen**: The preferred response (specific, safe, helpful)
# - **Rejected**: The response we want the model to avoid (vague, dismissive)
#
# The DPO loss pushes the model toward "chosen" and away from "rejected",
# relative to its starting behavior.

# %%
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
data_mod = importlib.import_module("data")
PREFERENCE_DATA = data_mod.PREFERENCE_DATA

console.print(f"  Loaded {len(PREFERENCE_DATA)} preference pairs")

# Show a few examples
table = Table(title="Sample Preference Pairs")
table.add_column("Prompt", max_width=30)
table.add_column("Chosen (preferred)", max_width=40, style="green")
table.add_column("Rejected (avoid)", max_width=40, style="red")

for ex in PREFERENCE_DATA[:3]:
    table.add_row(
        ex["prompt"][:30],
        ex["chosen"][:40] + "...",
        ex["rejected"][:40] + "...",
    )
console.print(table)

# %% [markdown]
# ---
# ## 5. Format for TRL's DPOTrainer
#
# TRL expects a HuggingFace Dataset with `prompt`, `chosen`, and `rejected`
# columns. We use the conversational format — each value is a list of
# message dicts.

# %%
from datasets import Dataset


def format_for_dpo(examples):
    """Convert our preference data to TRL's conversational format."""
    formatted = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for ex in examples:
        formatted["prompt"].append([{"role": "user", "content": ex["prompt"]}])
        formatted["chosen"].append([{"role": "assistant", "content": ex["chosen"]}])
        formatted["rejected"].append([{"role": "assistant", "content": ex["rejected"]}])
    return formatted


formatted = format_for_dpo(PREFERENCE_DATA)
train_dataset = Dataset.from_dict(formatted)

console.print(f"  Dataset: {len(train_dataset)} examples")
console.print(f"  Columns: {train_dataset.column_names}")

# %% [markdown]
# ---
# ## 6. Configure LoRA + DPO
#
# We use PEFT's LoRA config with TRL's DPOTrainer. Key settings:
#
# - **beta = 0.1**: Controls how far the model can deviate from its
#   starting behavior. Lower = more aggressive preference learning.
# - **LoRA rank = 16**: Higher than Lab 07's rank-8 because DPO needs
#   more capacity to learn subtle preference shifts.
# - **Learning rate = 5e-7**: Much lower than SFT (10x less) — DPO is
#   a fine adjustment, not a major behavior change.

# %%
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

N_STEPS = 100 if not is_ci() else 10
dpo_config = DPOConfig(
    output_dir="outputs/08-dpo-checkpoints",
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    learning_rate=5e-7,  # DPO uses much lower LR than SFT
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_steps=N_STEPS,
    logging_steps=max(1, N_STEPS // 10),
    save_steps=N_STEPS,  # save only at end
    remove_unused_columns=False,
    bf16=torch.cuda.is_available(),  # bf16 on CUDA, fp32 on MPS/CPU
    report_to="none",
)

console.print("  DPO config ready")
console.print(f"  beta={dpo_config.beta}, lr={dpo_config.learning_rate}")
console.print(f"  LoRA: r={lora_config.r}, target={lora_config.target_modules}")

# %% [markdown]
# ---
# ## 7. Train
#
# TRL's DPOTrainer handles everything:
# - Applies LoRA to the model automatically
# - Uses the base model (with LoRA disabled) as the reference
# - Computes the DPO loss on chosen vs rejected pairs
# - Logs training metrics

# %%
console.print("\n[bold]Starting DPO training...[/bold]\n")
t0 = time.time()

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # auto-handled: disables LoRA for ref
    args=dpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

train_result = trainer.train()
elapsed = time.time() - t0

console.print(f"\n  Training complete in {elapsed:.0f}s")

# Extract loss history from trainer logs
train_losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]

# %% [markdown]
# ---
# ## 8. Training Loss Curve

# %%
if train_losses:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, color="#b87333", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Logging Step")
    ax.set_ylabel("DPO Loss")
    ax.set_title("DPO Training Loss", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    show(fig, filename="08-dpo-loss-curve.png")

# %% [markdown]
# The DPO loss measures how well the model distinguishes chosen from
# rejected responses. As it decreases, the model increasingly prefers
# the chosen style (specific, safe, helpful) over the rejected style
# (vague, dismissive).

# %% [markdown]
# ---
# ## 9. After Alignment — Compare Responses
#
# Now let's ask the same questions and see how the responses changed.

# %%
# Merge LoRA for faster inference
model = trainer.model.merge_and_unload()
model.eval()

console.print("\n[bold]After DPO — Side-by-Side Comparison[/bold]\n")

for prompt in TEST_PROMPTS:
    after_response = generate(model, prompt)

    console.print(f"  [bold]Q:[/bold] {prompt}\n")
    console.print("  [dim]Before:[/dim]")
    console.print(f"    {baseline_responses[prompt][:250]}\n")
    console.print("  [green]After DPO:[/green]")
    console.print(f"    {after_response[:250]}\n")
    console.print("  " + "─" * 60 + "\n")

# %% [markdown]
# ---
# ## 10. Quantify the Shift
#
# Let's measure how much the model's preferences actually shifted
# by computing log-probabilities of chosen vs rejected responses.

# %%


def compute_response_logprob(mdl, prompt_text, response_text):
    """Compute the total log-probability of a response given a prompt."""
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        logits = mdl(ids).logits[0]  # (seq_len, vocab)

    # Log-probs for the response tokens only
    shift_logits = logits[prompt_len - 1 : -1]
    shift_labels = ids[0, prompt_len:]
    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze()
    return token_logprobs.sum().item()


# %%
# Measure preference margin on a few examples
n_eval = min(5, len(PREFERENCE_DATA))
margins = []

table = Table(title="Preference Margins After DPO")
table.add_column("Prompt", max_width=35)
table.add_column("log P(chosen)", justify="right")
table.add_column("log P(rejected)", justify="right")
table.add_column("Margin", justify="right")
table.add_column("Prefers")

for ex in PREFERENCE_DATA[:n_eval]:
    lp_chosen = compute_response_logprob(model, ex["prompt"], ex["chosen"])
    lp_rejected = compute_response_logprob(
        model,
        ex["prompt"],
        ex["rejected"],
    )
    margin = lp_chosen - lp_rejected
    margins.append(margin)
    prefers = "[green]chosen[/]" if margin > 0 else "[red]rejected[/]"
    table.add_row(
        ex["prompt"][:35],
        f"{lp_chosen:.1f}",
        f"{lp_rejected:.1f}",
        f"{margin:+.1f}",
        prefers,
    )

console.print(table)

avg_margin = np.mean(margins)
pct_correct = sum(1 for m in margins if m > 0) / len(margins)
console.print(f"\n  Average margin: [bold]{avg_margin:+.1f}[/bold]  (positive = prefers chosen)")
console.print(f"  Win rate: [bold]{pct_correct:.0%}[/bold]")

# %% [markdown]
# A positive margin means the model assigns higher probability to the
# chosen (detailed, safe) response than to the rejected (vague) one.
# After DPO training, we expect most or all margins to be positive.

# %% [markdown]
# ---
# ## 11. Save the Aligned Adapter

# %%
from microscale.viz import _output_dir

output_path = _output_dir()
adapter_path = output_path / "08-dpo-adapter"
trainer.save_model(str(adapter_path))

adapter_size = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()) / 1e6

console.print(f"\n  Adapter saved: {adapter_path}")
console.print(f"  Adapter size: [bold]{adapter_size:.1f} MB[/bold]")

# %% [markdown]
# ---
# ## What You Learned
#
# | Concept | Detail |
# |---|---|
# | DPO loss | Optimizes preferences without a reward model |
# | beta parameter | Controls deviation from base behavior |
# | Chosen vs rejected | Model learns response *style*, not just content |
# | Reference model | LoRA's base weights serve as the reference |
# | Preference margin | log P(chosen) - log P(rejected) quantifies alignment |
#
# ### Artifacts in `outputs/`
#
# | File | What it is |
# |------|-----------|
# | `08-dpo-loss-curve.png` | DPO training loss |
# | `08-dpo-adapter/` | Saved PEFT adapter (LoRA weights) |
#
# ### References
#
# - Rafailov et al., "Direct Preference Optimization" (2023, arXiv:2305.18290)
# - TRL Documentation: https://huggingface.co/docs/trl
# - Lab 07 (LoRA in 50 Lines) — the SFT step that precedes alignment
