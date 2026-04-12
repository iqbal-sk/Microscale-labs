# Microscale Labs

Hands-on labs for [Microscale Academy](https://www.microscale.academy/) — a field journal for Small Language Models.

Every lab produces a number you can't get from prose. Every lab runs on consumer hardware. Every lab leaves you with a reusable artifact.

## Quick Start

**Option A: Colab (zero setup)**

Click the Colab badge on any lab below.

**Option B: Local with uv (recommended)**

```bash
git clone https://github.com/user/Microscale.git
cd Microscale

# Auto-detect your platform (Mac/GPU/CPU) and install:
just setup-auto

# Or pick manually:
just setup          # Mac / CPU-only
just setup-cuda     # Linux + NVIDIA GPU (CUDA 12.4)
just setup-cuda126  # Linux + NVIDIA GPU (CUDA 12.6, newer drivers)
```

**Option C: Local with pip**

```bash
git clone https://github.com/user/Microscale.git
cd Microscale
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,dev,notebooks]"   # Mac / CPU
# or: pip install -e ".[cu124,dev,notebooks]"  # Linux + NVIDIA GPU
jupyter lab
```

**Option D: Run as scripts (no notebook required)**

```bash
just run 01
# or: uv run python labs/01-token-tax/lab.py
```

## Labs

| # | Lab | Act | CPU | GPU | Mac | Colab | Time |
|---|-----|-----|-----|-----|-----|-------|------|
| 01 | [The Token Tax](labs/01-token-tax/) | I | yes | — | — | yes | 30m |
| 02 | [Attention Under the Microscope](labs/02-attention-microscope/) | II | yes | faster | yes | yes | 60-90m |
| 03 | [Build a Transformer Block](labs/03-build-a-transformer/) | II | yes | — | yes | yes | 90-120m |
| 04 | [Model Autopsy](labs/04-model-autopsy/) | III | yes | — | — | yes | 45-60m |
| 05 | [The $1 Pretraining Run](labs/05-dollar-pretraining/) | IV | slow | yes | yes | T4 | 90-120m |
| 06 | [The Hallucination Probe](labs/06-hallucination-probe/) | V | yes | — | — | yes | 60-90m |
| 07 | [LoRA in 50 Lines](labs/07-lora-from-scratch/) | VI | slow | yes | yes | T4 | 60-90m |
| 08 | Your First DPO Alignment | VI | slow | yes | yes | T4 | 90m |
| 09 | Quantize It Yourself | VII | yes | — | — | yes | 90m |
| 10 | The Roofline Lab | VIII | no | yes | Metal | yes | 60-90m |
| 11 | KV Cache Calculator | VIII | partial | yes | yes | yes | 60m |
| 12 | The Inference Showdown | IX | yes | yes | yes | partial | 45-60m |

## Hardware Requirements

These labs are designed for consumer hardware:

- **Mac:** M-series with 16-32GB unified RAM (MPS acceleration auto-detected)
- **NVIDIA:** RTX 3060/4060+ with 8-12GB VRAM
- **Colab Free:** T4 GPU, 15GB VRAM
- **CPU:** Works for most labs (slower for training-heavy ones like 05, 07, 08)

## How Labs Work

Each lab is a **Jupytext percent-format `.py` file** — simultaneously a valid Python script and a Jupyter notebook source:

```
labs/01-token-tax/lab.py        ← you edit this (source of truth)
        │
        ├─→ python lab.py        ← run as script (outputs to outputs/)
        ├─→ VS Code / JupyterLab ← open as notebook (# %% cells supported natively)
        └─→ just sync            ← generates .ipynb for Colab
```

Labs import shared utilities from the `microscale` package — device detection, visualization helpers, model loading — so each lab stays focused on educational content. Adding a new lab is one file that imports `from microscale import ...`.

## Project Structure

```
microscale/              Shared reusable package
├── device.py            CPU / CUDA / MPS / MLX auto-detection
├── viz.py               Dual-output plotting (notebook + CLI)
├── models.py            Pinned model registry
├── metrics.py           Perplexity computation
├── attention.py         Attention extraction and visualization
├── transformer_block.py From-scratch transformer (matches Qwen3)
├── tiny_gpt.py          10M-param GPT-2 for pretraining labs
├── autopsy.py           Safetensors header parsing
├── lora.py              LoRA from scratch
├── cache.py             HuggingFace cache management
└── env.py               Environment detection (notebook/Colab/CI)

labs/                    Lab source files (.py, Jupytext percent format)
├── 01-token-tax/
├── 02-attention-microscope/
├── 03-build-a-transformer/
├── 04-model-autopsy/
├── 05-dollar-pretraining/
├── 06-hallucination-probe/
├── 07-lora-from-scratch/
└── ...

notebooks/               CI-generated .ipynb files (for Colab)
tests/                   42 unit tests across all modules
```

## Task Runner

Install [just](https://github.com/casey/just) (`brew install just` on Mac, `cargo install just` elsewhere):

```bash
just                    # list all commands
just setup-auto         # auto-detect platform, install deps
just run 01             # run lab 01 as a script
just lab                # open JupyterLab
just test               # run all 42 tests
just sync               # generate .ipynb from .py sources
just prefetch           # download all models for offline use
just lint               # lint all code with ruff
just fmt                # format all code with ruff
```

## API Keys

Lab 06 (Hallucination Probe) uses the OpenRouter API. Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys). The lab costs less than $0.01 to run.

## License

MIT
