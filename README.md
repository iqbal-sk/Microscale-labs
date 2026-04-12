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
uv sync --extra dev --extra notebooks
uv run jupyter lab
```

**Option C: Local with pip**

```bash
git clone https://github.com/user/Microscale.git
cd Microscale
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
jupyter lab
```

**Option D: Run as scripts (no notebook required)**

```bash
uv sync
uv run python labs/01-token-tax/lab.py
# or: just run 01
```

## Labs

| # | Lab | Act | CPU | GPU | Mac | Colab | Time |
|---|-----|-----|-----|-----|-----|-------|------|
| 01 | [The Token Tax](labs/01-token-tax/) | I | yes | — | — | yes | 30m |
| 02 | Attention Under the Microscope | II | yes | faster | yes | yes | 60-90m |
| 03 | Build a Transformer Block | II | yes | — | yes | yes | 90-120m |
| 04 | Model Autopsy | III | yes | — | — | yes | 45-60m |
| 05 | The $1 Pretraining Run | IV | slow | yes | yes | T4 | 90-120m |
| 06 | The Hallucination Probe | V | yes | faster | yes | yes | 60-90m |
| 07 | LoRA in 50 Lines | VI | slow | yes | yes | T4 | 60-90m |
| 08 | Your First DPO Alignment | VI | slow | yes | yes | T4 | 90m |
| 09 | Quantize It Yourself | VII | yes | — | — | yes | 90m |
| 10 | The Roofline Lab | VIII | no | yes | Metal | yes | 60-90m |
| 11 | KV Cache Calculator | VIII | partial | yes | yes | yes | 60m |
| 12 | The Inference Showdown | IX | yes | yes | yes | partial | 45-60m |

## Hardware Requirements

These labs target consumer hardware:
- **Mac:** M-series with 16-32GB unified RAM
- **NVIDIA:** RTX 3060/4060+ with 8-12GB VRAM
- **Colab Free:** T4 GPU, 15GB VRAM
- **CPU:** Works for most labs (slower for training-heavy ones)

## Task Runner

If you have [just](https://github.com/casey/just) installed:

```bash
just              # list commands
just setup        # install deps
just run 01       # run lab 01 as a script
just lab          # open JupyterLab
just test         # run all tests
just sync         # generate .ipynb from .py sources
just prefetch     # download all models for offline use
```

## Project Structure

```
microscale/       # Shared reusable package (device, viz, models, cache)
labs/             # Lab source files (.py, Jupytext percent format)
notebooks/        # Generated .ipynb files (for Colab)
tests/            # Unit tests + lab smoke tests
```

## How Labs Work

Each lab is a **Jupytext percent-format `.py` file** — a valid Python script that also opens as a notebook:

- **Run as a script:** `python labs/01-token-tax/lab.py` (outputs save to `outputs/`)
- **Open as a notebook:** VS Code, JupyterLab, and PyCharm all natively support `# %%` cells
- **Open in Colab:** CI generates `.ipynb` files with "Open in Colab" badges

Labs import shared utilities from the `microscale` package — device detection, visualization helpers, model loading — so each lab stays focused on its educational content.

## License

MIT
