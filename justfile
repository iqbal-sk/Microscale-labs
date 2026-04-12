# justfile — Microscale Labs task runner
# Run `just` or `just --list` to see all commands.

# Default: show available commands
default:
    @just --list

# ─── Setup ───────────────────────────────────────

# Install all dependencies (dev + notebooks)
setup:
    uv sync --extra dev --extra notebooks

# Install with MLX support (Mac only)
setup-apple:
    uv sync --extra dev --extra notebooks --extra apple

# Install with llama.cpp inference tools
setup-inference:
    uv sync --extra dev --extra notebooks --extra inference

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
