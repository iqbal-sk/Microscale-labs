# Microscale Labs

## Project Overview

Hands-on labs for Microscale Academy (microscale.academy). A shared `microscale/` Python package provides reusable utilities; each lab in `labs/XX-name/lab.py` is a Jupytext percent-format file that runs as both a Python script and a Jupyter notebook.

## Development Guidelines

### Git Conventions

- **Commit messages must never mention AI, Claude, or any AI assistant.** No `Co-Authored-By` AI lines.
- Write commit messages in imperative mood (e.g., "Add feature" not "Added feature").
- Keep the first line under 72 characters.
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `ci:`.

### Code Style

- Follow existing patterns in the codebase.
- Prefer clarity over cleverness.
- Ruff for linting and formatting (configured in pyproject.toml).

### Lab Conventions

- Labs are Jupytext percent format `.py` files in `labs/XX-name/lab.py`.
- Each lab imports reusable utilities from `microscale/` — never duplicate shared logic.
- Every lab must run as both `python lab.py` (script) and as a notebook.
- First cell of every lab handles Colab install + device detection.
- Use `microscale.viz.show()` for all figures — it routes to inline or file output automatically.

### Testing

- `uv run pytest tests/ -v` — unit tests
- `uv run python labs/01-token-tax/lab.py` — smoke test a lab

## Commands

```bash
just setup-auto     # Auto-detect platform and install
just setup          # Install for Mac / CPU
just setup-cuda     # Install for Linux + NVIDIA GPU
just run 01         # Run lab 01 as a script
just lab            # Open JupyterLab
just test           # Run all tests
just lint           # Lint all code
just fmt            # Format all code
just sync           # Generate .ipynb from .py sources
just prefetch       # Download all models for offline use
just strip          # Strip notebook outputs
```

## Architecture

- `microscale/` — shared package (device detection, viz, model loading, cache)
- `labs/XX-name/lab.py` — Jupytext percent format, source of truth
- `notebooks/` — CI-generated .ipynb files for Colab
- `tests/` — pytest unit tests + nbmake smoke tests
