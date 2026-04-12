# tests/test_env.py
from microscale.env import is_notebook, is_colab, is_ci, runtime_context


def test_is_notebook_returns_false_in_script():
    """Running under pytest is not a notebook."""
    assert is_notebook() is False


def test_is_colab_returns_false_locally():
    assert is_colab() is False


def test_is_ci_reflects_env(monkeypatch):
    monkeypatch.delenv("MICROSCALE_CI", raising=False)
    assert is_ci() is False
    monkeypatch.setenv("MICROSCALE_CI", "1")
    assert is_ci() is True


def test_runtime_context_returns_dict():
    ctx = runtime_context()
    assert "environment" in ctx
    assert "is_notebook" in ctx
    assert "is_colab" in ctx
    assert "is_ci" in ctx
    assert ctx["environment"] in ("script", "notebook", "colab")
