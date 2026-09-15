from unittest.mock import patch

from rag_app.models.providers import ModelProvider


def test_configure_registers_local_embedding_and_llm() -> None:
    settings = type(
        "SettingsStub",
        (),
        {"embedding_model": "local-embedding", "llm_model": "test-llm"},
    )()
    embedding = object()
    llm = object()

    with (
        patch("rag_app.models.providers.LlamaSettings") as llama_settings,
        patch("rag_app.models.providers.HuggingFaceEmbedding", return_value=embedding) as embed_cls,
        patch("rag_app.models.providers.GoogleGenAI", return_value=llm) as llm_cls,
    ):
        ModelProvider(settings).configure()

    embed_cls.assert_called_once_with(model_name="local-embedding")
    llm_cls.assert_called_once_with(model="test-llm")
    assert llama_settings.embed_model is embedding
    assert llama_settings.llm is llm
