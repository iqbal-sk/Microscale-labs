# microscale/env.py
"""Environment detection: notebook vs script, Colab vs local, CI vs interactive."""

import os


def is_notebook() -> bool:
    """Return True if running inside any Jupyter/Colab notebook kernel.

    Detects:
        - Jupyter (ZMQInteractiveShell)
        - Google Colab (Shell class, or google.colab importable)
        - VS Code notebooks (ZMQInteractiveShell)
    """
    # Fast path: Colab has its own detection
    if is_colab():
        return True

    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False
        # Any IPython instance that has a 'kernel' attribute is a notebook
        # (as opposed to a plain IPython terminal shell)
        return hasattr(ipython, "kernel")
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


def get_secret(name: str, prompt_if_missing: bool = True) -> str | None:
    """Get a secret from the appropriate source for the current environment.

    Priority order:
    1. Environment variable (works everywhere)
    2. `.env` file (local development)
    3. Colab userdata secrets (Colab notebooks)
    4. Interactive prompt (if prompt_if_missing=True and we're in a TTY/notebook)

    Args:
        name: The secret name (e.g., "OPENROUTER_API_KEY").
        prompt_if_missing: Ask user interactively if not found.

    Returns:
        The secret value, or None if not found and not prompted.
    """
    # 1. Already in environment?
    value = os.environ.get(name)
    if value:
        return value

    # 2. Try loading .env file (local dev)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        value = os.environ.get(name)
        if value:
            return value
    except ImportError:
        pass

    # 3. Colab secrets (the key icon in the left sidebar)
    if is_colab():
        try:
            from google.colab import userdata

            value = userdata.get(name)
            if value:
                return value
        except Exception:
            # userdata.SecretNotFoundError or similar
            pass

    # 4. Prompt user interactively
    if prompt_if_missing:
        try:
            import getpass

            value = getpass.getpass(f"Enter {name}: ")
            if value:
                os.environ[name] = value  # cache for subsequent calls
                return value
        except Exception:
            pass

    return None


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
