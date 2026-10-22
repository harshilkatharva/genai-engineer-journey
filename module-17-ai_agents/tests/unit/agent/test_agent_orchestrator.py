"""Unit tests for BoundedAgent orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from genai_core import (
    ApprovalDecision,
    RetrievalService,
    ToolBudget,
    ToolLoopStatus,
)

from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.core.config import AgentConfig
from research_and_support_agent.hitl.postgres_store import PostgresApprovalStore
from research_and_support_agent.memory.manager import AgentMemoryManager
from research_and_support_agent.models import CustomerAccount, ReversalAuditRecord, SupportTicket
from research_and_support_agent.telemetry.pricing import PricingCalculator


@pytest.mark.asyncio
async def test_agent_preturn_memory_injection(
    mock_llm_service: MagicMock,
    memory_manager: AgentMemoryManager,
    mock_retrieval_service: RetrievalService,
    approval_store: PostgresApprovalStore,
    agent_config: AgentConfig,
    sample_accounts: dict[str, CustomerAccount],
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> None:
    """Verify that recalling facts for user_id injects them into the agent system context."""
    # Pre-seed memory for user U-100
    await memory_manager.save_longterm_memory(
        user_id="U-100",
        content="User prefers email communications and dark mode theme.",
        category="preference",
    )

    agent = BoundedAgent(
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

    result = await agent.run(query="What are my settings?", user_id="U-100")
    assert result.status == ToolLoopStatus.COMPLETED

    # Verify complete_tool was called with messages containing memory facts
    called_messages = mock_llm_service.complete_tool.call_args[1]["messages"]
    system_content = next(m.content for m in called_messages if m.role == "system")
    assert "User prefers email communications and dark mode theme." in system_content


@pytest.mark.asyncio
async def test_agent_budget_override(
    mock_llm_service: MagicMock,
    memory_manager: AgentMemoryManager,
    mock_retrieval_service: RetrievalService,
    approval_store: PostgresApprovalStore,
    agent_config: AgentConfig,
    sample_accounts: dict[str, CustomerAccount],
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> None:
    """Verify run-level ToolBudget takes precedence over default configuration."""
    agent = BoundedAgent(
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

    custom_budget = ToolBudget(max_steps=2, max_tokens=500)
    result = await agent.run(query="Check ticket T-101", budget=custom_budget)
    assert result.status == ToolLoopStatus.COMPLETED


@pytest.mark.asyncio
async def test_agent_resume_without_pending_request(
    mock_llm_service: MagicMock,
    memory_manager: AgentMemoryManager,
    mock_retrieval_service: RetrievalService,
    approval_store: PostgresApprovalStore,
    agent_config: AgentConfig,
    sample_accounts: dict[str, CustomerAccount],
    sample_tickets: dict[str, SupportTicket],
    audit_log: list[ReversalAuditRecord],
) -> None:
    """Verify resuming when no approval request is pending returns a clean error."""
    agent = BoundedAgent(
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

    decision = ApprovalDecision(approved=True, reason="Valid")
    with pytest.raises(ValueError, match="not found"):
        await agent.resume(
            request_id="req_does_not_exist",
            decision=decision,
            actor_id="reviewer",
            actor_role="support_team",
        )
