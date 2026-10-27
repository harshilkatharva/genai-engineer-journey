"""Live integration tests for genai-core LLMService and MemoryService against real providers and PostgreSQL."""

from __future__ import annotations

import os

import pytest
from genai_core import (
    EmbeddingService,
    GoogleProvider,
    LLMManagerRequest,
    LLMService,
    LocalEmbeddingProvider,
    PostgresMemoryStore,
)
from genai_core.services.memory_service import MemoryService
from pydantic import BaseModel, Field


class TechComparisonSchema(BaseModel):
    """Schema for validating structured Gemini completion."""

    technology_a: str = Field(description="Name of first technology")
    technology_b: str = Field(description="Name of second technology")
    primary_tradeoff: str = Field(description="Core tradeoff between them")


@pytest.fixture
def live_llm_service() -> LLMService:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not configured for live test.")
    model = os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash-lite")
    provider = GoogleProvider(api_key=api_key, model=model)
    return LLMService(providers={"google": provider}, default_provider="google")


@pytest.fixture
def live_memory_service() -> MemoryService:
    db_url = os.getenv(
        "DATABASE_CONNECTION_CONVERSATION_URL",
        "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
    )
    emb_provider = LocalEmbeddingProvider(model="sentence-transformers/all-MiniLM-L6-v2")
    emb_service = EmbeddingService(provider=emb_provider)
    mem_store = PostgresMemoryStore(connection_string=db_url, dimensions=384)
    return MemoryService(store=mem_store, embedding_service=emb_service)


@pytest.mark.asyncio
async def test_live_llm_service_structured_completion(live_llm_service: LLMService) -> None:
    """Verify live structured response completion with Gemini Flash Lite."""
    prompt = (
        "Compare Redis and PostgreSQL for caching. Extract their names and the primary tradeoff."
    )
    res = await live_llm_service.complete_structure(
        prompt=prompt,
        response_schema=TechComparisonSchema,
    )

    assert res.error is None
    assert res.structured_response is not None
    assert isinstance(res.structured_response, TechComparisonSchema)
    assert (
        "redis" in res.structured_response.technology_a.lower()
        or "redis" in res.structured_response.technology_b.lower()
    )
    assert len(res.structured_response.primary_tradeoff) > 10


@pytest.mark.asyncio
async def test_live_llm_service_text_completion(live_llm_service: LLMService) -> None:
    """Verify live plain text completion via LLMManagerRequest."""
    req = LLMManagerRequest(prompt="Answer in exactly three words: What is the capital of France?")
    res = await live_llm_service.complete(req)

    assert res.error is None
    assert res.text is not None
    assert "paris" in res.text.lower()


@pytest.mark.asyncio
async def test_live_postgres_memory_service_recall(live_memory_service: MemoryService) -> None:
    """Verify live semantic memory recall against PostgreSQL memory_lt_records."""
    # Query preference for seeded user U-200
    results = await live_memory_service.recall(
        user_id="U-200",
        query="visual theme preference",
        top_k=2,
    )

    assert len(results) >= 1
    top_record = results[0].record
    assert "dark mode" in top_record.content.lower()


@pytest.mark.asyncio
async def test_live_postgres_episodic_memory_record(live_memory_service: MemoryService) -> None:
    """Verify episodic action logging into PostgreSQL memory_ep_episodes."""
    episode = await live_memory_service.episodic.record(
        user_id="U-200",
        action_type="integration_test_action",
        parameters={"test": True, "target": "database"},
        result_summary="Integration test action succeeded",
    )

    assert episode is not None
    assert episode.user_id == "U-200"
    assert episode.action_type == "integration_test_action"
