# justfile — Microscale Labs task runner
# Run `just` or `just --list` to see all commands.

# Default: show available commands
default:
    @just --list

# ─── Setup ───────────────────────────────────────

# Install for Mac or CPU-only (default)
setup:
    uv sync --extra cpu --extra dev --extra notebooks

# Install for Linux with NVIDIA GPU (CUDA 12.4)
setup-cuda:
    uv sync --extra cu124 --extra dev --extra notebooks

# Install for Linux with NVIDIA GPU (CUDA 12.6 — newer drivers)
setup-cuda126:
    uv sync --extra cu126 --extra dev --extra notebooks

# Install with MLX support (Mac only)
setup-apple:
    uv sync --extra cpu --extra apple --extra dev --extra notebooks

# Install with llama.cpp inference tools
setup-inference:
    uv sync --extra cpu --extra dev --extra notebooks --extra inference

# Auto-detect platform and install accordingly
setup-auto:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected — installing with CUDA 12.4 support"
        uv sync --extra cu124 --extra dev --extra notebooks
    elif [[ "$(uname)" == "Darwin" ]]; then
        echo "macOS detected — installing with MPS support"
        uv sync --extra cpu --extra dev --extra notebooks
    else
        echo "No GPU detected — installing CPU-only"
        uv sync --extra cpu --extra dev --extra notebooks
    fi
    echo ""
    uv run python -c "from microscale import device_summary; print(device_summary())"

# ─── Labs ────────────────────────────────────────

# Open JupyterLab
lab:
    uv run jupyter lab

# Run a specific lab as a script: just run 01
run lab_num:
    uv run python labs/{{lab_num}}-*/lab.py

# ─── Notebooks ───────────────────────────────────

# Sync all .py labs to .ipynb notebooks
sync:
    @for f in labs/*/lab.py; do \
        uv run jupytext --to ipynb --output notebooks/$(basename $(dirname "$f")).ipynb "$f"; \
    done
    @echo "Synced all notebooks to notebooks/"

# Sync a specific lab: just sync-lab 01
sync-lab lab_num:
    uv run jupytext --to ipynb --output notebooks/$(basename $(dirname $(ls labs/{{lab_num}}-*/lab.py))).ipynb labs/{{lab_num}}-*/lab.py

# ─── Quality ─────────────────────────────────────

# Run all tests
test:
    uv run pytest tests/ -v

# Test a specific lab notebook
test-lab lab_num:
    uv run pytest --nbmake labs/{{lab_num}}-*/lab.py --nbmake-timeout=600

# Lint all code
lint:
    uv run ruff check microscale/ labs/ tests/
    uv run ruff format --check microscale/ labs/ tests/

# Format all code
fmt:
    uv run ruff check --fix microscale/ labs/ tests/
    uv run ruff format microscale/ labs/ tests/

# Strip notebook outputs before commit
strip:
    find labs -name "*.ipynb" -exec uv run nbstripout {} \;
    find notebooks -name "*.ipynb" -exec uv run nbstripout {} \;

# ─── Models ──────────────────────────────────────

# Prefetch all registered models for offline use
prefetch:
    uv run python -c "from microscale.models import MODEL_REGISTRY; from huggingface_hub import snapshot_download; [print(f'Downloading {n}...') or snapshot_download(m['repo'], revision=m['revision']) for n, m in MODEL_REGISTRY.items()]"
