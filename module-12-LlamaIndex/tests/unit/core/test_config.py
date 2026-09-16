from pathlib import Path

from rag_app.core.config import Settings


def test_settings_defaults_use_local_models_and_persistent_storage() -> None:
    settings = Settings()

    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_dimension == 384
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert settings.data_dir == Path(settings.data_dir)
    assert settings.index_storage_dir.name == "storage"


def test_settings_accept_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("APP_NAME", "test-rag")
    monkeypatch.setenv("CHUNK_SIZE", "256")

    settings = Settings()

    assert settings.app_name == "test-rag"
    assert settings.chunk_size == 256
