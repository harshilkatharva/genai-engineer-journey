import pytest

from rag_app.config import Settings


def test_settings_default_values() -> None:
    settings = Settings()

    assert settings.app_name == "Semantic Search Engine"
    assert settings.app_version == "0.1.0"
    assert settings.debug is False
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.data_directory == "src/rag_app/user_data/data"
    assert settings.chunking_strategy == "sentence"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.embedding_batch_size == 100
    assert settings.embedding_cost_per_million_tokens == 0.0
    assert settings.default_top_k == 5
    assert settings.max_top_k == 100
    assert settings.log_directory == "src/rag_app/user_data/logs"
    assert settings.log_level == "INFO"


def test_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "custom-model",
    )
    monkeypatch.setenv(
        "EMBEDDING_COST_PER_MILLION_TOKENS",
        "0.25",
    )

    settings = Settings()

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_cost_per_million_tokens == 0.25


@pytest.mark.parametrize(
    "value",
    [0, -1],
)
def test_settings_reject_invalid_chunk_size(value: int) -> None:
    with pytest.raises(ValueError):
        Settings(chunk_size=value)


def test_settings_reject_negative_overlap() -> None:
    with pytest.raises(ValueError):
        Settings(chunk_overlap=-1)


def test_settings_reject_invalid_api_port() -> None:
    with pytest.raises(ValueError):
        Settings(api_port=0)
