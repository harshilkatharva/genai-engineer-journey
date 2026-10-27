"""Live integration tests for Module 17 BoundedAgent against live tools and PostgreSQL."""

from __future__ import annotations

import os

import psycopg
import pytest
from genai_core import (
    EmbeddingService,
    GoogleProvider,
    LLMService,
    LocalEmbeddingProvider,
    PGVectorStoreAdapter,
    PostgresMemoryStore,
    RetrievalService,
)
from genai_core.services.memory_service import MemoryService
from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.core.config import AgentConfig
from research_and_support_agent.data.db_store import (
    load_customer_accounts,
    load_support_tickets,
)
from research_and_support_agent.hitl.postgres_store import PostgresApprovalStore
from research_and_support_agent.memory.manager import AgentMemoryManager


@pytest.fixture
async def live_bounded_agent() -> BoundedAgent:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not configured for live test.")
    db_url = os.getenv(
        "DATABASE_CONNECTION_CONVERSATION_URL",
        "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
    )
    model = os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash-lite")

    accounts = await load_customer_accounts(db_url)
    tickets = await load_support_tickets(db_url)

    provider = GoogleProvider(api_key=api_key, model=model)
    llm = LLMService(providers={"google": provider}, default_provider="google")
    emb = EmbeddingService(
        provider=LocalEmbeddingProvider(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    mem_store = PostgresMemoryStore(connection_string=db_url, dimensions=384)
    memory_service = MemoryService(store=mem_store, embedding_service=emb)
    memory_manager = AgentMemoryManager(memory_service=memory_service)
    approval_store = PostgresApprovalStore(connection_string=db_url)

    pg_vector = PGVectorStoreAdapter(
        dimensions=384,
        connection_factory=lambda: psycopg.AsyncConnection.connect(db_url),
        table_name="document_chunks",
    )
    retrieval = RetrievalService(store=pg_vector, embedding_service=emb)

    return BoundedAgent(
        llm_service=llm,
        memory_manager=memory_manager,
        retrieval_service=retrieval,
        approval_store=approval_store,
        account_store=accounts,
        ticket_store=tickets,
        config=AgentConfig(default_model=model, default_max_steps=5),
    )


@pytest.mark.asyncio
async def test_live_bounded_agent_customer_account_lookup(live_bounded_agent: BoundedAgent) -> None:
    """Verify BoundedAgent executes lookup_customer_account tool and returns verified details."""
    result = await live_bounded_agent.run(
        "What is the account tier and balance for customer C-101?"
    )

    assert result.final_text is not None
    assert "gold" in result.final_text.lower()
    assert "$100" in result.final_text or "100.00" in result.final_text


@pytest.mark.asyncio
async def test_live_bounded_agent_ticket_status_lookup(live_bounded_agent: BoundedAgent) -> None:
    """Verify BoundedAgent executes ticket_status tool against PostgreSQL tickets."""
    result = await live_bounded_agent.run("What is the current subject and status of ticket T-101?")

    assert result.final_text is not None
    text_lower = result.final_text.lower()
    assert "t-101" in text_lower or "billing" in text_lower or "duplicate" in text_lower


@pytest.mark.asyncio
async def test_live_bounded_agent_knowledge_base_search(live_bounded_agent: BoundedAgent) -> None:
    """Verify BoundedAgent searches vector knowledge base document_chunks."""
    result = await live_bounded_agent.run(
        "What is the company warranty return policy for hardware?"
    )

    assert result.final_text is not None
    text_lower = result.final_text.lower()
    assert "warranty" in text_lower or "30" in text_lower or "return" in text_lower
