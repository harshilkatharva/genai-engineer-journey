from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from genai_core.embedding.google_provider import GoogleEmbeddingProvider
from genai_core.embedding.protocol import EmbeddingProvider
from genai_core.exceptions import EmbeddingDimensionError, EmbeddingProviderError


def test_google_provider_implements_protocol() -> None:
    mock_client = MagicMock()
    provider = GoogleEmbeddingProvider(client=mock_client)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.provider_name == "google"

    assert provider.default_model == "gemini-embedding-001"
    assert provider.model == "gemini-embedding-001"
    assert provider.native_dimensions == 768


def test_google_dimension_validation() -> None:
    mock_client = MagicMock()
    provider = GoogleEmbeddingProvider(client=mock_client, model="gemini-embedding-001")

    # Supports dimensions <= 768
    provider.validate_dimensions(768)
    provider.validate_dimensions(256)

    with pytest.raises(EmbeddingDimensionError):
        provider.validate_dimensions(1024)

    with pytest.raises(EmbeddingDimensionError):
        provider.validate_dimensions(0)


@pytest.mark.asyncio
async def test_google_embed_batch_success() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Create mock ContentEmbedding items
    emb0 = MagicMock(values=[0.1] * 256)
    emb1 = MagicMock(values=[0.2] * 256)
    mock_response.embeddings = [emb0, emb1]

    mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

    provider = GoogleEmbeddingProvider(
        client=mock_client,
        model="text-embedding-005",
        dimensions=256,
    )

    results = await provider.embed_batch(["alpha", "beta"], normalize=True)

    assert len(results) == 2
    assert len(results[0]) == 256
    # Check L2 normalization
    norm = math.sqrt(sum(x * x for x in results[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    mock_client.aio.models.embed_content.assert_called_once()
    call_kwargs = mock_client.aio.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-005"
    assert call_kwargs["contents"] == ["alpha", "beta"]
    assert call_kwargs["config"].output_dimensionality == 256


@pytest.mark.asyncio
async def test_google_embed_batch_error_wrapping() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        side_effect=RuntimeError("Google Quota Exceeded")
    )

    provider = GoogleEmbeddingProvider(client=mock_client)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await provider.embed_batch(["sample query"])

    assert "Google Quota Exceeded" in str(exc_info.value)
    assert exc_info.value.provider == "google"
