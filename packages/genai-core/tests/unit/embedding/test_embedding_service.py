from __future__ import annotations

import asyncio
import math
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_core.embedding.local_provider import LocalEmbeddingProvider
from genai_core.embedding.protocol import EmbeddingProvider
from genai_core.exceptions import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
    MissingExtraDependencyError,
)
from genai_core.models import EmbeddingRequest, EmbeddingResponse
from genai_core.observability import (
    ObservabilityEvent,
    configure_observability,
)
from genai_core.services import EmbeddingService


class MockEmbeddingProvider:
    """Mock provider conforming to EmbeddingProvider protocol for testing."""

    def __init__(self, model_name: str = "test-model", dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "MockEmbeddingProvider"

    @property
    def default_model(self) -> str:
        return self.model_name

    @property
    def native_dimensions(self) -> int:
        return self.dimension

    def validate_dimensions(self, dimensions: int) -> None:
        if dimensions != self.dimension:
            raise EmbeddingDimensionError(
                model=self.model_name,
                requested=dimensions,
                supported=self.dimension,
            )

    async def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        self.calls.append(texts)
        result: list[list[float]] = []
        for text in texts:
            val = float(len(text))
            vec = [val] * self.dimension
            if normalize:
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                result.append([x / norm for x in vec])
            else:
                result.append(vec)
        return result


class FailingEmbeddingProvider:
    """Mock provider that simulates an underlying failure."""

    @property
    def provider_name(self) -> str:
        return "FailingEmbeddingProvider"

    @property
    def default_model(self) -> str:
        return "failing-model"

    @property
    def native_dimensions(self) -> int:
        return 384

    def validate_dimensions(self, dimensions: int) -> None:
        pass

    async def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> list[list[float]]:
        raise RuntimeError("Inference engine crashed")


# --- Protocol Tests ---


def test_mock_provider_implements_protocol() -> None:
    provider = MockEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)


# --- Initialization & Default Tests ---


def test_service_initialization_defaults() -> None:
    # Uses mock provider to avoid requiring sentence_transformers
    provider = MockEmbeddingProvider("all-MiniLM-L6-v2", 384)
    service = EmbeddingService(provider=provider)

    assert service.provider is provider
    assert service.model == "all-MiniLM-L6-v2"
    assert service.dimensions == 384
    assert service.batch_size == 32


def test_service_dimension_validation_success() -> None:
    provider = MockEmbeddingProvider("all-MiniLM-L6-v2", 384)
    service = EmbeddingService(provider=provider, dimensions=384)
    assert service.dimensions == 384


def test_service_dimension_validation_failure() -> None:
    provider = MockEmbeddingProvider("all-MiniLM-L6-v2", 384)
    with pytest.raises(EmbeddingDimensionError) as exc_info:
        EmbeddingService(provider=provider, dimensions=768)

    err = exc_info.value
    assert err.model == "all-MiniLM-L6-v2"
    assert err.requested == 768
    assert "does not support output dimension 768" in str(err)


# --- Input Validation Tests ---


@pytest.mark.asyncio
async def test_embed_empty_list_raises_value_error() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())
    with pytest.raises(ValueError, match="Input cannot be empty"):
        await service.embed(input=[])


@pytest.mark.asyncio
async def test_embed_empty_string_raises_value_error() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())
    with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
        await service.embed(input="")


@pytest.mark.asyncio
async def test_embed_whitespace_string_raises_value_error() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())
    with pytest.raises(ValueError, match="Input text cannot be empty or whitespace only"):
        await service.embed(input="   \n\t  ")


@pytest.mark.asyncio
async def test_embed_list_with_whitespace_item_raises_value_error() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())
    with pytest.raises(
        ValueError, match="Input text at index 1 cannot be empty or whitespace only"
    ):
        await service.embed(input=["valid text", "   "])


@pytest.mark.asyncio
async def test_embed_invalid_batch_size_raises_value_error() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        await service.embed(input=["hello"], batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        await service.embed(input=["hello"], batch_size=-10)


# --- Internal Batching and Order Preservation Tests ---


@pytest.mark.asyncio
async def test_internal_batching_and_order_preservation() -> None:
    provider = MockEmbeddingProvider("test-model", 384)
    service = EmbeddingService(provider=provider, batch_size=10)

    # 25 items -> batches should be [10, 10, 5]
    inputs = [f"document-content-{i:03d}" for i in range(25)]

    response = await service.embed(input=inputs, batch_size=10)

    assert isinstance(response, EmbeddingResponse)
    assert response.model == "test-model"
    assert response.dimensions == 384
    assert response.provider == "MockEmbeddingProvider"
    assert len(response.embeddings) == 25
    assert response.usage is None  # Local embeddings return None

    # Check that provider was called in 3 separate batches
    assert len(provider.calls) == 3
    assert len(provider.calls[0]) == 10
    assert len(provider.calls[1]) == 10
    assert len(provider.calls[2]) == 5

    # Check 1-to-1 order preservation
    for i, embedding in enumerate(response.embeddings):
        assert len(embedding) == 384
        expected_len = float(len(inputs[i]))
        expected_unnorm = [expected_len] * 384
        expected_norm = math.sqrt(sum(x * x for x in expected_unnorm))
        expected_val = expected_len / expected_norm
        assert math.isclose(embedding[0], expected_val, rel_tol=1e-5)


@pytest.mark.asyncio
async def test_embed_single_string_input() -> None:
    provider = MockEmbeddingProvider("test-model", 384)
    service = EmbeddingService(provider=provider)

    response = await service.embed(input="single string query")

    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == 384
    assert len(provider.calls) == 1
    assert provider.calls[0] == ["single string query"]


# --- Convenience Helpers Tests ---


@pytest.mark.asyncio
async def test_embed_query_convenience_helper() -> None:
    provider = MockEmbeddingProvider("test-model", 384)
    service = EmbeddingService(provider=provider)

    vec = await service.embed_query("how to index documents")

    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.asyncio
async def test_embed_documents_convenience_helper() -> None:
    provider = MockEmbeddingProvider("test-model", 384)
    service = EmbeddingService(provider=provider)

    docs = ["first document", "second document"]
    matrix = await service.embed_documents(docs)

    assert isinstance(matrix, list)
    assert len(matrix) == 2
    assert len(matrix[0]) == 384
    assert len(matrix[1]) == 384


@pytest.mark.asyncio
async def test_embed_with_embedding_request_object() -> None:
    provider = MockEmbeddingProvider("test-model", 384)
    service = EmbeddingService(provider=provider)

    req = EmbeddingRequest(input="explicit request object")
    resp = await service.embed(req)

    assert isinstance(resp, EmbeddingResponse)
    assert len(resp.embeddings) == 1
    assert len(resp.embeddings[0]) == 384


# --- Provider Error and Observability Tests ---


@pytest.mark.asyncio
async def test_provider_failure_raises_embedding_provider_error() -> None:
    failing_provider = FailingEmbeddingProvider()
    service = EmbeddingService(provider=failing_provider)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await service.embed(input=["test query"])

    assert "Inference engine crashed" in str(exc_info.value)
    assert exc_info.value.provider == "FailingEmbeddingProvider"


@pytest.mark.asyncio
async def test_embedding_observability_events() -> None:
    captured_events: list[tuple[str, dict[str, Any]]] = []

    def log_hook(name: str, payload: dict[str, Any]) -> None:
        captured_events.append((name, payload))

    configure_observability()

    with patch("genai_core.services.embedding_service.safe_log") as mock_safe_log:
        provider = MockEmbeddingProvider("test-model", 384)
        service = EmbeddingService(provider=provider)

        await service.embed(input=["hello world"])

        # Should log START and SUCCESS
        assert mock_safe_log.call_count == 2
        events = [call.kwargs.get("event") for call in mock_safe_log.call_args_list]
        assert events[0] == ObservabilityEvent.EMBEDDING_REQUEST_START
        assert events[1] == ObservabilityEvent.EMBEDDING_REQUEST_SUCCESS


# --- LocalEmbeddingProvider Specific Tests ---


def test_local_provider_missing_dependency() -> None:
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        provider = LocalEmbeddingProvider()
        with pytest.raises(MissingExtraDependencyError) as exc_info:
            provider._get_model()

        err_msg = str(exc_info.value)
        assert "sentence-transformers" in err_msg
        assert "genai-core[local]" in err_msg


def test_local_provider_dimension_validation() -> None:
    provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    # Native dimension is 384
    provider.validate_dimensions(384)

    with pytest.raises(EmbeddingDimensionError):
        provider.validate_dimensions(768)


@pytest.mark.asyncio
async def test_local_provider_embed_batch_with_mocked_st() -> None:
    provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    class MockVector:
        def __init__(self, data: list[float]) -> None:
            self.data = data

        def tolist(self) -> list[float]:
            return self.data

    mock_st_model = MagicMock()
    mock_st_model.encode.return_value = [
        MockVector([0.1] * 384),
        MockVector([0.2] * 384),
    ]

    with patch.object(provider, "_get_model", return_value=mock_st_model):
        results = await provider.embed_batch(["text 1", "text 2"])

        assert len(results) == 2
        assert len(results[0]) == 384
        mock_st_model.encode.assert_called_once_with(
            ["text 1", "text 2"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# --- Hosted Providers & Deadline Tests ---


@pytest.mark.asyncio
async def test_embedding_service_with_openai_provider_reports_usage() -> None:
    from genai_core.embedding.openai_provider import OpenAIEmbeddingProvider

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(index=0, embedding=[0.1] * 512)]
    mock_resp.usage = MagicMock(prompt_tokens=42, total_tokens=42)
    mock_client.embeddings.create = AsyncMock(return_value=mock_resp)

    openai_provider = OpenAIEmbeddingProvider(
        client=mock_client,
        model="text-embedding-3-small",
        dimensions=512,
    )
    service = EmbeddingService(provider=openai_provider, dimensions=512)

    response = await service.embed(input="openai query")

    assert response.provider == "openai"
    assert response.model == "text-embedding-3-small"
    assert response.dimensions == 512
    # Verify honest usage reporting: hosted providers report actual usage
    assert response.usage == {"prompt_tokens": 42, "total_tokens": 42}


@pytest.mark.asyncio
async def test_embedding_service_overall_deadline_breach() -> None:
    class ExtremelySlowProvider:
        @property
        def provider_name(self) -> str:
            return "SlowProvider"

        @property
        def default_model(self) -> str:
            return "slow-model"

        @property
        def native_dimensions(self) -> int:
            return 384

        def validate_dimensions(self, dimensions: int) -> None:
            pass

        async def embed_batch(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
            await asyncio.sleep(0.1)
            return [[0.1] * 384 for _ in texts]

    slow_provider = ExtremelySlowProvider()
    from genai_core.exceptions import EmbeddingTimeoutError
    from genai_core.resilience import RetryPolicy

    # Overall deadline of 0.04s, but batch takes 0.1s
    service = EmbeddingService(
        provider=slow_provider,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(timeout_seconds=0.04),
    )

    with pytest.raises(EmbeddingTimeoutError):
        await service.embed(input=["text 1", "text 2"])
