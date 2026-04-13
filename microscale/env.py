# microscale/env.py
"""Environment detection: notebook vs script, Colab vs local, CI vs interactive."""

import os


def is_notebook() -> bool:
    """Return True if running inside a Jupyter/Colab notebook kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def is_ci() -> bool:
    """Return True if MICROSCALE_CI=1 is set (used to reduce workload in CI)."""
    return os.environ.get("MICROSCALE_CI", "").strip() in ("1", "true", "yes")


def runtime_context() -> dict:
    """Return a summary dict of the current runtime environment."""
    if is_colab():
        env = "colab"
    elif is_notebook():
        env = "notebook"
    else:
        env = "script"

    return {
        "environment": env,
        "is_notebook": is_notebook(),
        "is_colab": is_colab(),
        "is_ci": is_ci(),
    }
