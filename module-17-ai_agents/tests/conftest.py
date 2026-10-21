"""Shared test fixtures for Module 17."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genai_core import (
    DocumentChunk,
    InMemoryMemoryStore,
    InMemoryVectorStore,
    LLMManagerResponse,
    MemoryService,
    RetrievalService,
)

from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.config import AgentConfig
from research_and_support_agent.hitl.postgres_store import PostgresApprovalStore
from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.models import (
    CustomerAccount,
    ReversalAuditRecord,
    SupportTicket,
    TicketStatus,
)
from research_and_support_agent.telemetry.pricing import PricingCalculator


@pytest.fixture
def agent_config(tmp_path: Any) -> AgentConfig:
    """Fixture providing isolated test configuration."""
    return AgentConfig(
        transcripts_dir=tmp_path / "transcripts",
        data_dir=tmp_path / "data",
        default_max_steps=5,
        default_max_tokens=2000,
        default_max_cost_usd=0.01,
        default_timeout_seconds=5.0,
        default_max_repeated_calls=3,
    )


@pytest.fixture
def sample_accounts() -> dict[str, CustomerAccount]:
    """Sample customer accounts store."""
    return {
        "C-101": CustomerAccount(
            customer_id="C-101",
            name="Alice Smith",
            email="alice@example.com",
            tier="Gold",
            balance_usd=100.0,
            active_tickets=["T-101"],
            risk_flag=False,
        ),
        "C-102": CustomerAccount(
            customer_id="C-102",
            name="Bob Jones",
            email="bob@example.com",
            tier="Standard",
            balance_usd=0.0,
            active_tickets=["T-102"],
            risk_flag=False,
        ),
    }


@pytest.fixture
def sample_tickets() -> dict[str, SupportTicket]:
    """Sample support tickets store."""
    return {
        "T-101": SupportTicket(
            ticket_id="T-101",
            customer_id="C-101",
            title="Broken screen on arrival",
            description="Item delivered with shattered display.",
            status=TicketStatus.OPEN,
            assigned_team="tier1_support",
            priority="high",
        ),
        "T-102": SupportTicket(
            ticket_id="T-102",
            customer_id="C-102",
            title="Billing inquiry",
            description="Clarification on renewal discount.",
            status=TicketStatus.OPEN,
            assigned_team="tier1_support",
            priority="medium",
        ),
    }


@pytest.fixture
def audit_log() -> list[ReversalAuditRecord]:
    """In-memory audit log for reversible actions."""
    return []


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock embedding service returning 4-dimensional unit vector."""
    mock_emb = MagicMock()
    mock_emb.dimensions = 4
    mock_emb.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0])
    return mock_emb


@pytest.fixture
def mock_memory_service(mock_embedding_service: MagicMock) -> MemoryService:
    """In-memory MemoryService from genai-core."""
    store = InMemoryMemoryStore()
    return MemoryService(store=store, embedding_service=mock_embedding_service)


@pytest.fixture
def memory_manager(mock_memory_service: MemoryService) -> AgentMemoryManager:
    """AgentMemoryManager fixture backed by in-memory MemoryService."""
    return AgentMemoryManager(memory_service=mock_memory_service)


@pytest.fixture
def mock_retrieval_service(mock_embedding_service: MagicMock) -> RetrievalService:
    """RetrievalService backed by InMemoryVectorStore with seeded policy chunks."""
    store = InMemoryVectorStore(dimensions=4)
    # Seed sample policy chunks with dummy 4-dim unit vectors
    chunks = [
        DocumentChunk(
            id="chunk_warr",
            content="Hardware warranty returns are accepted within 30 days of delivery.",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"source": "kb/hardware_policy.md"},
        ),
        DocumentChunk(
            id="chunk_vip",
            content="Gold and VIP tier members receive free expedited shipping on replacements.",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={"source": "kb/vip_perks.md"},
        ),
    ]
    store._upsert_sync(chunks)
    return RetrievalService(store=store, embedding_service=mock_embedding_service)


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Mock LLMService simulating deterministic tool calls or responses."""
    svc = MagicMock()
    # Default behavior: model responds with final answer directly
    resp = LLMManagerResponse(
        text="Default mock answer",
        model="gemini-1.5-flash",
        tool_calls=[],
        usage={"input_tokens": 50, "output_tokens": 20},
    )
    svc.complete_tool = AsyncMock(return_value=resp)
    return svc


@pytest.fixture
def approval_store() -> PostgresApprovalStore:
    """Approval store running in in-memory mode for unit tests."""
    return PostgresApprovalStore()


@pytest.fixture
def agent(
    mock_llm_service: MagicMock,
    memory_manager: AgentMemoryManager,
    mock_retrieval_service: RetrievalService,
    approval_store: PostgresApprovalStore,
    agent_config: AgentConfig,
    sample_accounts: dict[str, CustomerAccount],
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> BoundedAgent:
    """Fully wired BoundedAgent test fixture."""
    return BoundedAgent(
        llm_service=mock_llm_service,
        memory_manager=memory_manager,
        retrieval_service=mock_retrieval_service,
        approval_store=approval_store,
        config=agent_config,
        account_store=sample_accounts,
        ticket_store=sample_tickets,
        audit_log=audit_log,
        pricing_calculator=PricingCalculator(),
    )
