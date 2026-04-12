# tests/test_models.py
from microscale.models import MODEL_REGISTRY, get_model_info, list_models
from microscale.cache import setup_cache, cache_status


def test_registry_has_entries():
    assert len(MODEL_REGISTRY) > 0


def test_get_model_info_known_model():
    info = get_model_info("qwen3-0.6b")
    assert info is not None
    assert "repo" in info
    assert "revision" in info
    assert info["repo"].startswith("Qwen/")


def test_get_model_info_unknown_returns_none():
    assert get_model_info("nonexistent-model-xyz") is None


def test_list_models_returns_names():
    names = list_models()
    assert isinstance(names, list)
    assert "qwen3-0.6b" in names


def test_setup_cache_returns_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    path = setup_cache()
    assert path.exists()


def test_cache_status_returns_dict():
    status = cache_status()
    assert "cache_dir" in status
    assert "offline_mode" in status
