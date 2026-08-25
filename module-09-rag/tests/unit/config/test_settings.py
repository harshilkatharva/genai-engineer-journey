import pytest

from rag_app.core import Settings


def test_settings_default_values() -> None:
    settings = Settings()

    assert settings.app_name == "Semantic Search Engine"
    assert settings.app_version == "0.1.21"
    assert settings.debug is False
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.data_directory == "user_data/data"
    assert settings.chunking_strategy == "sentence"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.default_embedding_model == "all-MiniLM-L6-v2"
    assert settings.embedding_batch_size == 100
    assert settings.embedding_cost_per_million_tokens == 0.0
    assert settings.canidate_default_top_k == 5
    assert settings.canidate_max_top_k == 100
    assert settings.re_ranker_default_top_k == 5
    assert settings.re_ranker_max_top_k == 20
    assert settings.log_directory == "logs"
    assert settings.log_level == "INFO"


def test_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv(
        "DEFAULT_EMBEDDING_MODEL",
        "custom-model",
    )
    monkeypatch.setenv(
        "EMBEDDING_COST_PER_MILLION_TOKENS",
        "0.25",
    )

    settings = Settings()

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.default_embedding_model == "custom-model"
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
