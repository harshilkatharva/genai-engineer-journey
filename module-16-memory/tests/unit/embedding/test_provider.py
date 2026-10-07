from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from memory_system.embedding import provider
from memory_system.embedding.provider import SentenceTransformerEmbeddings


@pytest.mark.asyncio
async def test_embed_rejects_whitespace_and_runs_sync_work_off_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = SentenceTransformerEmbeddings("model", 2)
    to_thread = AsyncMock(return_value=[0.25, 0.75])
    monkeypatch.setattr(provider.asyncio, "to_thread", to_thread)

    with pytest.raises(ValueError, match="empty text"):
        await embeddings.embed(" \n")
    assert await embeddings.embed("hello") == [0.25, 0.75]
    to_thread.assert_awaited_once_with(embeddings._embed_sync, "hello")


def test_embed_sync_loads_model_once_and_converts_vector_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    class Vector:
        def tolist(self) -> list[int]:
            return [1, 2]

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            loaded.append(model_name)

        def encode(self, text: str, normalize_embeddings: bool) -> Vector:
            assert text == "hello"
            assert normalize_embeddings is True
            return Vector()

    fake_module = ModuleType("sentence_transformers")
    fake_module.__dict__["SentenceTransformer"] = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    embeddings = SentenceTransformerEmbeddings("local-model", 2)

    assert embeddings._embed_sync("hello") == [1.0, 2.0]
    assert embeddings._embed_sync("hello") == [1.0, 2.0]
    assert loaded == ["local-model"]


def test_embed_sync_rejects_wrong_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    class Vector:
        def tolist(self) -> list[float]:
            return [1.0]

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            pass

        def encode(self, text: str, normalize_embeddings: bool) -> Vector:
            return Vector()

    fake_module = ModuleType("sentence_transformers")
    fake_module.__dict__["SentenceTransformer"] = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(ValueError, match="1 dimensions; configured for 2"):
        SentenceTransformerEmbeddings("local-model", 2)._embed_sync("hello")
