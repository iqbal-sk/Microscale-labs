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
# # Lab 06: The Hallucination Probe
#
# **Act V — Where They Break** | CPU only (API-based) | ~60–90 minutes
#
# ---
#
# ### What you will learn
#
# 1. **Design** a factual question bank with 25 common-knowledge and
#    25 long-tail questions, each with verified ground-truth answers
# 2. **Query** small language models via the OpenRouter API — the same
#    interface used for production LLM applications
# 3. **Score** model responses with deterministic multi-alias matching
# 4. **Measure** the hallucination rate per category and see that it
#    increases with question rarity — exactly as theory predicts
# 5. **Understand** the Kalai-Vempala bound: why hallucination is a
#    mathematical consequence of finite training data, not a fixable bug
#
# ---
#
# ### The idea
#
# The Microscale Academy lesson on hallucination says: "Small models
# hallucinate more on rare facts because those facts appeared fewer times
# in training data." That is a theoretical claim. Today you will measure
# it yourself.
#
# You will ask a small model 50 factual questions — half common knowledge,
# half obscure — and watch the hallucination rate climb as questions get
# rarer. The pattern you observe will match the Kalai-Vempala bound from
# STOC 2024: hallucination rate is bounded below by the fraction of facts
# that appear rarely in training data.
#
# ---
#
# ### Requirements
#
# This lab uses the **OpenRouter API** (openrouter.ai). You need:
# - An API key set as `OPENROUTER_API_KEY` environment variable
# - Free-tier models are available, so no payment is needed

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

from microscale import apply_style, device_summary, is_ci, show

apply_style()
print(device_summary())

# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.table import Table

console = Console()

# %% [markdown]
# ---
# ## 2. Connect to OpenRouter
#
# OpenRouter provides an OpenAI-compatible API that lets you access many
# models through one endpoint. We use the `openai` Python package with
# a custom base URL.

# %%
# Load .env file if it exists (supports OPENROUTER_API_KEY in .env)
from dotenv import load_dotenv

load_dotenv()  # loads from .env in current dir or parent dirs

# Check common env var names for OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get(
    "OPENROUTER_API_SECRET", ""
)

if not OPENROUTER_API_KEY:
    console.print(
        "[bold red]No OpenRouter API key found.[/]\n"
        "  Create a .env file in the project root with:\n"
        "    OPENROUTER_API_KEY=sk-or-your-key-here\n"
        "  Get a free key at: https://openrouter.ai/keys"
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Small model — $0.05/M tokens, 50 questions costs ~$0.0001
# Free-tier (:free suffix) models often hit rate limits, so we
# use the paid tier which has much higher limits.
# Change this to test other models.
MODEL = "meta-llama/llama-3.2-3b-instruct"
console.print(f"  Model: [bold]{MODEL}[/bold]")

# %% [markdown]
# ---
# ## 3. The Question Bank
#
# We split 50 questions into two groups:
#
# - **Common knowledge** (25 questions): facts that appear thousands of
#   times in any web corpus — capitals, basic science, famous dates
# - **Long-tail** (25 questions): facts that appear rarely — obscure
#   dates, specific numbers, niche geography
#
# Each question has one or more acceptable answer aliases for
# deterministic scoring.

# %%
QUESTIONS = {
    "common": [
        {"q": "What is the capital of France?", "answers": ["paris"]},
        {"q": "What is the chemical symbol for water?", "answers": ["h2o"]},
        {"q": "In what year did World War II end?", "answers": ["1945"]},
        {"q": "What planet is closest to the Sun?", "answers": ["mercury"]},
        {"q": "What is the largest ocean on Earth?", "answers": ["pacific"]},
        {"q": "Who wrote Romeo and Juliet?", "answers": ["shakespeare", "william shakespeare"]},
        {"q": "What is the boiling point of water in Celsius?", "answers": ["100"]},
        {
            "q": "What gas do plants absorb from the atmosphere?",
            "answers": ["carbon dioxide", "co2"],
        },
        {"q": "What is the capital of Japan?", "answers": ["tokyo"]},
        {"q": "How many continents are there?", "answers": ["7", "seven"]},
        {
            "q": "What is the speed of light in km/s (approximately)?",
            "answers": ["300000", "300,000", "299792"],
        },
        {
            "q": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "leonardo da vinci"],
        },
        {"q": "What is the capital of Australia?", "answers": ["canberra"]},
        {"q": "What year did the Berlin Wall fall?", "answers": ["1989"]},
        {"q": "What is the largest planet in our solar system?", "answers": ["jupiter"]},
        {
            "q": "What is the currency of the United Kingdom?",
            "answers": ["pound", "gbp", "pound sterling"],
        },
        {"q": "What element has atomic number 1?", "answers": ["hydrogen"]},
        {
            "q": "Who was the first person to walk on the Moon?",
            "answers": ["armstrong", "neil armstrong"],
        },
        {"q": "What is the square root of 144?", "answers": ["12"]},
        {"q": "In which country is the Great Wall located?", "answers": ["china"]},
        {"q": "What is the capital of Germany?", "answers": ["berlin"]},
        {"q": "What is the freezing point of water in Fahrenheit?", "answers": ["32"]},
        {
            "q": "What language has the most native speakers?",
            "answers": ["mandarin", "chinese", "mandarin chinese"],
        },
        {"q": "How many legs does a spider have?", "answers": ["8", "eight"]},
        {
            "q": "What is the tallest mountain in the world?",
            "answers": ["everest", "mount everest"],
        },
    ],
    "long_tail": [
        {"q": "What is the capital of Vanuatu?", "answers": ["port vila"]},
        {"q": "What year was the Treaty of Westphalia signed?", "answers": ["1648"]},
        {"q": "What is the atomic number of Ruthenium?", "answers": ["44"]},
        {
            "q": "Who discovered the element Hafnium?",
            "answers": ["coster", "hevesy", "coster and hevesy", "dirk coster", "george de hevesy"],
        },
        {"q": "What is the deepest lake in Africa?", "answers": ["tanganyika", "lake tanganyika"]},
        {"q": "In what year was the Kelvin temperature scale proposed?", "answers": ["1848"]},
        {"q": "What is the capital of Bhutan?", "answers": ["thimphu"]},
        {"q": "What is the half-life of Carbon-14 in years?", "answers": ["5730"]},
        {
            "q": "Who was the first woman to win a Nobel Prize in Physics?",
            "answers": ["marie curie", "curie"],
        },
        {"q": "What is the largest desert in Asia?", "answers": ["gobi"]},
        {"q": "In what year was the Rosetta Stone discovered?", "answers": ["1799"]},
        {"q": "What is the capital of Suriname?", "answers": ["paramaribo"]},
        {"q": "What is the atomic mass of Neon (rounded)?", "answers": ["20"]},
        {
            "q": "Who invented the Mercator projection?",
            "answers": ["mercator", "gerardus mercator", "gerhard kremer"],
        },
        {"q": "What is the longest river in Australia?", "answers": ["murray", "murray river"]},
        {"q": "In what year did the Ottoman Empire officially end?", "answers": ["1922"]},
        {"q": "What is the capital of Eritrea?", "answers": ["asmara"]},
        {
            "q": "What is the melting point of gold in Celsius?",
            "answers": ["1064", "1,064", "1063", "1,063"],
        },
        {"q": "Who wrote 'The Wealth of Nations'?", "answers": ["adam smith", "smith"]},
        {"q": "How many bones are in the adult human body?", "answers": ["206"]},
        {"q": "What is the capital of Laos?", "answers": ["vientiane"]},
        {"q": "What year was the Suez Canal opened?", "answers": ["1869"]},
        {"q": "What element has the highest melting point?", "answers": ["tungsten", "carbon"]},
        {"q": "What is the smallest country in Africa by area?", "answers": ["seychelles"]},
        {"q": "In what year was Pluto reclassified as a dwarf planet?", "answers": ["2006"]},
    ],
}

console.print(
    f"  Question bank: {len(QUESTIONS['common'])} common + "
    f"{len(QUESTIONS['long_tail'])} long-tail = "
    f"{len(QUESTIONS['common']) + len(QUESTIONS['long_tail'])} total"
)

# %% [markdown]
# ---
# ## 4. Query the Model
#
# We use greedy decoding (temperature=0) for reproducible results. Each
# response is limited to 30 tokens — enough for a short factual answer.
#
# Rate limiting: we add a delay between requests and retry on 429 errors
# with exponential backoff.

# %%
RATE_LIMIT_DELAY = 3.0  # seconds between requests
MAX_RETRIES = 3


def query_model(question: str, model: str = MODEL) -> str:
    """Ask a factual question with retry logic for rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer with just the fact. Be brief."},
                    {"role": "user", "content": question},
                ],
                max_tokens=30,
                temperature=0,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "[EMPTY]"
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait = RATE_LIMIT_DELAY * (2**attempt)
                console.print(f"    [dim]Rate limited, waiting {wait:.0f}s...[/dim]")
                time.sleep(wait)
                continue
            return f"[ERROR: {e}]"
    return "[ERROR: rate limit exceeded after retries]"


def score_response(response: str, acceptable: list[str]) -> str:
    """Score a response against acceptable answers.

    Returns: 'correct', 'incorrect', or 'error'
    """
    if response.startswith("[ERROR"):
        return "error"
    # Normalize: lowercase, strip punctuation/whitespace, remove commas
    resp_normalized = response.lower().strip().replace(",", "")
    for alias in acceptable:
        alias_normalized = alias.lower().replace(",", "")
        if alias_normalized in resp_normalized:
            return "correct"
    return "incorrect"


# %% [markdown]
# Let's run the probe. This takes a few minutes due to rate limiting.

# %%
results = {"common": [], "long_tail": []}

# Limit questions in CI mode
n_per_category = 3 if is_ci() else 25

for category in ["common", "long_tail"]:
    questions = QUESTIONS[category][:n_per_category]
    label = "Common Knowledge" if category == "common" else "Long-Tail"
    console.print(f"\n  [bold]Probing {label} ({len(questions)} questions)...[/bold]")

    for i, item in enumerate(questions):
        response = query_model(item["q"])
        verdict = score_response(response, item["answers"])
        results[category].append(
            {
                "question": item["q"],
                "expected": item["answers"][0],
                "response": response,
                "verdict": verdict,
            }
        )

        # Show progress
        icon = {"correct": "[green]OK[/]", "incorrect": "[red]WRONG[/]", "error": "[yellow]ERR[/]"}[
            verdict
        ]
        console.print(f"    {i + 1:2d}. {icon}  Q: {item['q'][:50]:50s}  A: {response[:40]}")

        time.sleep(RATE_LIMIT_DELAY)

# %% [markdown]
# ---
# ## 5. Results

# %%
# Compute hallucination rates
for category in ["common", "long_tail"]:
    label = "Common Knowledge" if category == "common" else "Long-Tail"
    entries = results[category]
    n_total = len(entries)
    n_correct = sum(1 for e in entries if e["verdict"] == "correct")
    n_wrong = sum(1 for e in entries if e["verdict"] == "incorrect")
    n_error = sum(1 for e in entries if e["verdict"] == "error")

    table = Table(title=f"{label} Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total questions", str(n_total))
    table.add_row("Correct", f"[green]{n_correct}[/] ({n_correct / n_total:.0%})")
    table.add_row(
        "Incorrect (hallucinated)",
        f"[red]{n_wrong}[/] ({n_wrong / n_total:.0%})",
    )
    if n_error:
        table.add_row("Errors", f"[yellow]{n_error}[/]")
    table.add_row(
        "Hallucination rate",
        f"[bold]{n_wrong / max(n_total - n_error, 1):.0%}[/bold]",
    )
    console.print(table)

# %%
# Detailed results table
for category in ["common", "long_tail"]:
    label = "Common" if category == "common" else "Long-Tail"
    table = Table(title=f"{label} — Detailed Responses")
    table.add_column("#", justify="right", width=3)
    table.add_column("Question", max_width=40)
    table.add_column("Expected")
    table.add_column("Model Response", max_width=40)
    table.add_column("Verdict")

    for i, entry in enumerate(results[category], 1):
        color = {"correct": "green", "incorrect": "red", "error": "yellow"}[entry["verdict"]]
        table.add_row(
            str(i),
            entry["question"][:40],
            entry["expected"],
            entry["response"][:40],
            f"[{color}]{entry['verdict']}[/]",
        )
    console.print(table)

# %% [markdown]
# ---
# ## 6. Visualize the Hallucination Pattern

# %%
# Compute rates
common_entries = results["common"]
longtail_entries = results["long_tail"]

common_valid = [e for e in common_entries if e["verdict"] != "error"]
longtail_valid = [e for e in longtail_entries if e["verdict"] != "error"]

common_hall_rate = sum(1 for e in common_valid if e["verdict"] == "incorrect") / max(
    len(common_valid), 1
)
longtail_hall_rate = sum(1 for e in longtail_valid if e["verdict"] == "incorrect") / max(
    len(longtail_valid), 1
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart: hallucination rate by category
ax = axes[0]
categories = ["Common\nKnowledge", "Long-Tail\nFacts"]
rates = [common_hall_rate, longtail_hall_rate]
colors = ["#4a7c74", "#8b3a3a"]
bars = ax.bar(categories, rates, color=colors, width=0.5, edgecolor="#1a1f3a", linewidth=1.5)
for bar, rate in zip(bars, rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{rate:.0%}",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
ax.set_ylabel("Hallucination Rate")
ax.set_title("Hallucination Rate by Question Category")
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color="#6b7091", linestyle="--", alpha=0.3)
ax.grid(True, alpha=0.2, axis="y")

# Scatter: per-question results
ax = axes[1]
all_qs = []
for cat, label, color in [
    ("common", "Common", "#4a7c74"),
    ("long_tail", "Long-Tail", "#8b3a3a"),
]:
    for i, entry in enumerate(results[cat]):
        x_val = i if cat == "common" else i + len(results["common"]) + 2
        is_wrong = 1 if entry["verdict"] == "incorrect" else 0
        marker = "x" if is_wrong else "o"
        ax.scatter(x_val, is_wrong, color=color, marker=marker, s=80, alpha=0.7, zorder=5)

ax.set_yticks([0, 1])
ax.set_yticklabels(["Correct", "Hallucinated"])
ax.set_xlabel("Question Index")
ax.set_title("Per-Question Results")
ax.axvline(
    x=len(results["common"]) + 1,
    color="#6b7091",
    linestyle="--",
    alpha=0.3,
    label="Category boundary",
)
ax.text(len(results["common"]) / 2, 1.1, "Common", ha="center", fontsize=10, color="#4a7c74")
ax.text(
    len(results["common"]) + 2 + len(results["long_tail"]) / 2,
    1.1,
    "Long-Tail",
    ha="center",
    fontsize=10,
    color="#8b3a3a",
)
ax.set_ylim(-0.2, 1.3)

fig.suptitle(
    f"The Hallucination Probe — {MODEL.split('/')[-1]}",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="06-hallucination-rates.png")

# %% [markdown]
# ---
# ## 7. Response Analysis — Do Wrong Answers Look Different?
#
# Do hallucinated responses have telltale signs? Let's analyze response
# length by category and verdict.

# %%
# Compute response length statistics
all_lengths = {
    "common_correct": [],
    "common_wrong": [],
    "longtail_correct": [],
    "longtail_wrong": [],
}

for cat in ["common", "long_tail"]:
    cat_key = "common" if cat == "common" else "longtail"
    for entry in results[cat]:
        if entry["verdict"] == "error":
            continue
        key = f"{cat_key}_{'correct' if entry['verdict'] == 'correct' else 'wrong'}"
        all_lengths[key].append(len(entry["response"]))

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Panel 1: Response length distributions by category and verdict
ax = axes[0]
groups = [
    ("Common\nCorrect", all_lengths["common_correct"], "#4a7c74"),
    ("Common\nWrong", all_lengths["common_wrong"], "#8b3a3a"),
    ("Long-tail\nCorrect", all_lengths["longtail_correct"], "#5a7a3d"),
    ("Long-tail\nWrong", all_lengths["longtail_wrong"], "#b87333"),
]

positions = list(range(len(groups)))
labels = [g[0] for g in groups]
data = [g[1] if g[1] else [0] for g in groups]
colors_box = [g[2] for g in groups]

bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=True)
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
    patch.set_edgecolor("#1a1f3a")

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Response Length (characters)")
ax.set_title("Response Length by Category and Verdict")
ax.grid(True, alpha=0.3, axis="y")

# Panel 2: Detailed per-question error type
ax = axes[1]
common_verdicts = [e["verdict"] for e in results["common"]]
longtail_verdicts = [e["verdict"] for e in results["long_tail"]]

from collections import Counter

verdicts_common = Counter(common_verdicts)
verdicts_longtail = Counter(longtail_verdicts)

categories = ["Correct", "Incorrect", "Error"]
common_counts = [
    verdicts_common.get("correct", 0),
    verdicts_common.get("incorrect", 0),
    verdicts_common.get("error", 0),
]
longtail_counts = [
    verdicts_longtail.get("correct", 0),
    verdicts_longtail.get("incorrect", 0),
    verdicts_longtail.get("error", 0),
]

x = np.arange(len(categories))
width = 0.35
ax.bar(
    x - width / 2,
    common_counts,
    width,
    label="Common Knowledge",
    color="#4a7c74",
    edgecolor="#1a1f3a",
)
ax.bar(
    x + width / 2,
    longtail_counts,
    width,
    label="Long-Tail Facts",
    color="#8b3a3a",
    edgecolor="#1a1f3a",
)

for i, (c, lt) in enumerate(zip(common_counts, longtail_counts)):
    ax.text(i - width / 2, c + 0.3, str(c), ha="center", fontweight="bold")
    ax.text(i + width / 2, lt + 0.3, str(lt), ha="center", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Number of Questions")
ax.set_title("Verdict Breakdown by Category")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "How Hallucinations Differ from Correct Answers",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
show(fig, filename="06-response-analysis.png")

# Summary statistics
console.print("\n[bold]Response Length Analysis[/bold]")
for label, lens, _ in groups:
    if lens:
        mean_len = np.mean(lens)
        console.print(
            f"  {label.replace(chr(10), ' '):25s}  "
            f"n={len(lens):2d}  "
            f"mean={mean_len:5.0f} chars  "
            f"median={int(np.median(lens)):4d}"
        )

# %% [markdown]
# **Patterns to watch for:**
# - Wrong answers are sometimes *shorter* — the model gives up
# - Wrong answers are sometimes *longer* — the model rambles, generating
#   plausible-looking but incorrect detail (classic hallucination)
# - Correct common-knowledge answers tend to be crisp and confident
# - Long-tail wrong answers often contain invented numbers or names

# %% [markdown]
# ---
# ## 8. The Kalai-Vempala Bound
#
# In 2024, Kalai and Vempala proved a mathematical lower bound on LLM
# hallucination (STOC 2024, arXiv:2311.14648):
#
# > **For any calibrated language model, the hallucination rate on
# > factual questions is bounded below by the monofact rate — the
# > fraction of facts that appear only once in the training data.**
#
# The intuition comes from the Good-Turing estimator in statistics:
# the probability of encountering a *new* (never-seen) species is
# approximately equal to the fraction of species seen exactly once.
# Applied to language models: the probability of generating a *wrong*
# fact is bounded by the fraction of facts the model barely learned.
#
# ### What this means for our results
#
# - **Common knowledge** questions (capitals, basic science) appear
#   thousands of times in training data → low monofact rate → low
#   hallucination rate
# - **Long-tail** questions (obscure dates, niche geography) appear
#   rarely → high monofact rate → high hallucination rate
#
# The pattern you measured — more hallucination on rarer facts — is not
# a model defect. It is a mathematical consequence of the training data
# distribution. No amount of RLHF, prompt engineering, or fine-tuning
# can push hallucination below this bound for truly rare facts.

# %%
# Theoretical bound visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Our measured data points
measured_x = [1, 2]
measured_y = [common_hall_rate, longtail_hall_rate]
ax.bar(
    measured_x,
    measured_y,
    width=0.4,
    color=["#4a7c74", "#8b3a3a"],
    edgecolor="#1a1f3a",
    linewidth=1.5,
    label="Measured",
    zorder=5,
)

for x_pos, rate in zip(measured_x, measured_y):
    ax.text(x_pos, rate + 0.02, f"{rate:.0%}", ha="center", fontsize=12, fontweight="bold")

# Theoretical bound curve (illustrative)
x_theory = np.linspace(0.5, 2.5, 100)
# Monofact rate increases with question rarity
bound_curve = 0.05 + 0.4 * ((x_theory - 0.5) / 2) ** 1.5
ax.plot(
    x_theory,
    bound_curve,
    color="#b87333",
    linewidth=2,
    linestyle="--",
    label="Kalai-Vempala bound\n(illustrative)",
    alpha=0.8,
)
ax.fill_between(x_theory, bound_curve, 0, color="#b87333", alpha=0.08)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Common Knowledge\n(high frequency)", "Long-Tail Facts\n(low frequency)"])
ax.set_ylabel("Hallucination Rate")
ax.set_title(
    "Hallucination Rate vs Fact Frequency\n(Kalai & Vempala, STOC 2024)",
    fontsize=12,
)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis="y")

fig.tight_layout()
show(fig, filename="06-kalai-vempala-bound.png")

# %% [markdown]
# The shaded area represents the theoretical minimum hallucination rate —
# no model can consistently beat this bound on rare facts without having
# seen those facts sufficiently often in training.

# %% [markdown]
# ---
# ## What You Learned
#
# | Finding | Your evidence |
# |---|---|
# | Common facts have low hallucination | Measured rate on 25 well-known facts |
# | Rare facts have high hallucination | Measured rate on 25 obscure facts |
# | The gap matches theory | Kalai-Vempala: rate >= monofact fraction |
# | Hallucination is inevitable | Mathematical bound, not a bug |
# | Smaller models hallucinate more | 3B model vs what you would see on GPT-4 |
#
# ### Artifacts in `outputs/`
#
# | File | What it shows |
# |------|---------------|
# | `06-hallucination-rates.png` | Bar chart + per-question scatter |
# | `06-kalai-vempala-bound.png` | Measured vs theoretical bound |
#
# ### References
#
# - Kalai & Vempala, "Calibrated Language Models Must Hallucinate"
#   (STOC 2024, arXiv:2311.14648)
# - Kalai et al., "Why Language Models Hallucinate"
#   (arXiv:2509.04664, 2025)
