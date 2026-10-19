from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from genai_core.embedding.openai_provider import OpenAIEmbeddingProvider
from genai_core.embedding.protocol import EmbeddingProvider
from genai_core.exceptions import EmbeddingDimensionError, EmbeddingProviderError


def test_openai_provider_implements_protocol() -> None:
    mock_client = MagicMock()
    provider = OpenAIEmbeddingProvider(client=mock_client)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.provider_name == "openai"
    assert provider.default_model == "text-embedding-3-small"
    assert provider.native_dimensions == 1536


def test_openai_dimension_validation() -> None:
    mock_client = MagicMock()
    provider = OpenAIEmbeddingProvider(client=mock_client, model="text-embedding-3-small")

    # text-embedding-3-* allows dimensions <= native
    provider.validate_dimensions(1536)
    provider.validate_dimensions(512)

    with pytest.raises(EmbeddingDimensionError):
        provider.validate_dimensions(2048)

    with pytest.raises(EmbeddingDimensionError):
        provider.validate_dimensions(-1)


def test_openai_ada_dimension_validation() -> None:
    mock_client = MagicMock()
    provider = OpenAIEmbeddingProvider(client=mock_client, model="text-embedding-ada-002")

    # ada-002 only allows 1536
    provider.validate_dimensions(1536)

    with pytest.raises(EmbeddingDimensionError):
        provider.validate_dimensions(512)


@pytest.mark.asyncio
async def test_openai_embed_batch_success_and_usage() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Create mock embedding items with non-sequential return to test order sorting
    item1 = MagicMock(index=1, embedding=[0.2] * 512)
    item0 = MagicMock(index=0, embedding=[0.1] * 512)
    mock_response.data = [item1, item0]  # Out of order
    mock_response.usage = MagicMock(prompt_tokens=15, total_tokens=15)

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    provider = OpenAIEmbeddingProvider(
        client=mock_client,
        model="text-embedding-3-small",
        dimensions=512,
    )

    results = await provider.embed_batch(["text first", "text second"], normalize=True)

    # Verify order is strictly sorted by index: item0 then item1
    assert len(results) == 2
    assert len(results[0]) == 512
    # Check L2 normalization
    norm = math.sqrt(sum(x * x for x in results[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    # Check that usage was recorded
    assert provider.last_usage == {"prompt_tokens": 15, "total_tokens": 15}

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["text first", "text second"],
        dimensions=512,
    )


@pytest.mark.asyncio
async def test_openai_embed_batch_error_wrapping() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("OpenAI API Down"))

    provider = OpenAIEmbeddingProvider(client=mock_client)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await provider.embed_batch(["test text"])

    assert "OpenAI API Down" in str(exc_info.value)
    assert exc_info.value.provider == "openai"
