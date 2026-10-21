"""Tool registry construction and dependency binding for Module 17."""

from __future__ import annotations

from typing import Any

import httpx
from genai_core.services import RetrievalService
from genai_core.tools import RiskLevel, ToolRegistry, tool

from ..memory.manager import AgentMemoryManager
from ..models import CustomerAccount, ReversalAuditRecord, SupportTicket
from .irreversible import (
    delete_customer_data,
    delete_episodic_memory_all,
    delete_episodic_memory_episodes,
    delete_longterm_memory_all,
    delete_longterm_memory_specific,
    execute_refund,
    purge_expired,
)
from .read_only import (
    get_episodic_memory,
    get_longterm_memory,
    lookup_customer_account,
    search_knowledge_base,
    search_web,
    ticket_status,
)
from .reversible import (
    escalate_to_tier2,
    save_longterm_memory,
    update_ticket_status,
)


def create_agent_tool_registry(
    retrieval_service: RetrievalService | None = None,
    memory_manager: AgentMemoryManager | None = None,
    account_store: dict[str, CustomerAccount] | None = None,
    ticket_store: dict[str, SupportTicket] | None = None,
    audit_log: list[ReversalAuditRecord] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ToolRegistry:
    """Build and register all 16 domain tools with clean schemas and appropriate risk levels."""
    registry = ToolRegistry()

    # Default in-memory stores if not provided
    accounts = account_store if account_store is not None else {}
    tickets = ticket_store if ticket_store is not None else {}
    audit = audit_log if audit_log is not None else []

    # -------------------------------------------------------------
    # 1. READ-ONLY TOOLS (RiskLevel.READ_ONLY)
    # -------------------------------------------------------------

    @tool(
        name="search_knowledge_base",
        description="Search internal documentation and policy manuals for company guidelines and technical rules.",
        risk_level=RiskLevel.READ_ONLY,
        tags=["research", "kb", "books", "policy"],
    )
    async def tool_search_kb(query: str, top_k: int = 3) -> str:
        return await search_knowledge_base(
            query=query,
            top_k=top_k,
            retrieval_service=retrieval_service,
        )

    @tool(
        name="search_web",
        description="Search public web pages for technical documentation, external guidelines, updates and any web pages content.",
        risk_level=RiskLevel.READ_ONLY,
        tags=["research", "web"],
    )
    async def tool_search_web(query: str, max_results: int = 3) -> str:
        return await search_web(
            query=query,
            max_results=max_results,
            client=http_client,
        )

    @tool(
        name="get_longterm_memory",
        description="Retrieve relevant long-term memory records (preferences, profile facts) for the authorized user.",
        risk_level=RiskLevel.READ_ONLY,
        tags=["memory", "read"],
    )
    async def tool_get_longterm(user_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await get_longterm_memory(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_manager=memory_manager,
        )

    @tool(
        name="get_episodic_memory",
        description="Retrieve past action history, parameters, and execution outcomes for the authorized user.",
        risk_level=RiskLevel.READ_ONLY,
        tags=["memory", "read"],
    )
    async def tool_get_episodic(
        user_id: str, action_type: str | None = None, limit: int = 15
    ) -> list[dict[str, Any]]:
        return await get_episodic_memory(
            user_id=user_id,
            action_type=action_type,
            limit=limit,
            memory_manager=memory_manager,
        )

    @tool(
        name="ticket_status",
        description="Retrieve the current status, details, and assignment for a support ticket.",
        risk_level=RiskLevel.READ_ONLY,
        tags=["support", "tickets"],
    )
    async def tool_ticket_status(ticket_id: str) -> dict[str, Any]:
        return await ticket_status(ticket_id=ticket_id, ticket_store=tickets)

    @tool(
        name="lookup_customer_account",
        description="Retrieve customer account information, tier, balance, and open ticket count.",
        risk_level=RiskLevel.READ_ONLY,
        tags=["support", "accounts"],
    )
    async def tool_lookup_customer(customer_id: str) -> dict[str, Any]:
        return await lookup_customer_account(customer_id=customer_id, account_store=accounts)

    # -------------------------------------------------------------
    # 2. REVERSIBLE TOOLS (RiskLevel.REVERSIBLE_WRITE)
    # -------------------------------------------------------------

    @tool(
        name="save_longterm_memory",
        description="Store or update a user fact or preference in persistent long-term memory.",
        risk_level=RiskLevel.REVERSIBLE_WRITE,
        tags=["memory", "write"],
    )
    async def tool_save_longterm(
        user_id: str, content: str, category: str = "fact"
    ) -> dict[str, Any]:
        return await save_longterm_memory(
            user_id=user_id,
            content=content,
            category=category,
            memory_manager=memory_manager,
        )

    @tool(
        name="update_ticket_status",
        description="Update ticket status while retaining sufficient information to reverse the change.",
        risk_level=RiskLevel.REVERSIBLE_WRITE,
        tags=["support", "tickets"],
    )
    async def tool_update_ticket(
        ticket_id: str, new_status: str, comment: str = ""
    ) -> dict[str, Any]:
        return await update_ticket_status(
            ticket_id=ticket_id,
            new_status=new_status,
            comment=comment,
            ticket_store=tickets,
            audit_log=audit,
        )

    @tool(
        name="escalate_to_tier2",
        description="Escalate a support ticket to Tier-2 engineers with supported de-escalation mechanism.",
        risk_level=RiskLevel.REVERSIBLE_WRITE,
        tags=["support", "tickets"],
    )
    async def tool_escalate_tier2(
        ticket_id: str, reason: str, urgency: str = "medium"
    ) -> dict[str, Any]:
        return await escalate_to_tier2(
            ticket_id=ticket_id,
            reason=reason,
            urgency=urgency,
            ticket_store=tickets,
            audit_log=audit,
        )

    # -------------------------------------------------------------
    # 3. IRREVERSIBLE TOOLS (RiskLevel.IRREVERSIBLE)
    # -------------------------------------------------------------

    @tool(
        name="delete_longterm_memory_all",
        description="Permanently delete all long-term memories belonging to the authorized user. Requires user confirmation.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["memory", "irreversible"],
    )
    async def tool_delete_lt_all(user_id: str) -> dict[str, Any]:
        return await delete_longterm_memory_all(user_id=user_id, memory_manager=memory_manager)

    @tool(
        name="delete_longterm_memory_specific",
        description="Permanently delete selected long-term memory record by ID. Requires user confirmation.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["memory", "irreversible"],
    )
    async def tool_delete_lt_specific(user_id: str, record_id: str) -> dict[str, Any]:
        return await delete_longterm_memory_specific(
            user_id=user_id, record_id=record_id, memory_manager=memory_manager
        )

    @tool(
        name="delete_episodic_memory_all",
        description="Permanently delete all episodic records belonging to the authorized user. Requires user confirmation.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["memory", "irreversible"],
    )
    async def tool_delete_ep_all(user_id: str) -> dict[str, Any]:
        return await delete_episodic_memory_all(user_id=user_id, memory_manager=memory_manager)

    @tool(
        name="delete_episodic_memory_episodes",
        description="Permanently delete selected episodic records by ID. Requires user confirmation.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["memory", "irreversible"],
    )
    async def tool_delete_ep_episodes(user_id: str, episode_ids: list[str]) -> dict[str, Any]:
        return await delete_episodic_memory_episodes(
            user_id=user_id, episode_ids=episode_ids, memory_manager=memory_manager
        )

    @tool(
        name="purge_expired",
        description="Remove only expired memory records. Never deletes active memory data. Requires support-team approval.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["memory", "maintenance", "irreversible"],
    )
    async def tool_purge_expired(user_id: str | None = None) -> dict[str, Any]:
        return await purge_expired(user_id=user_id, memory_manager=memory_manager)

    @tool(
        name="execute_refund",
        description="Execute a customer support refund. Gated by support-team approval and protected by episodic idempotency.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["finance", "support", "irreversible"],
    )
    async def tool_execute_refund(
        customer_id: str, amount_usd: float, reason: str
    ) -> dict[str, Any]:
        return await execute_refund(
            customer_id=customer_id,
            amount_usd=amount_usd,
            reason=reason,
            account_store=accounts,
            memory_manager=memory_manager,
        )

    @tool(
        name="delete_customer_data",
        description="Permanently delete customer personal data within authorized scope. Requires support-team approval.",
        risk_level=RiskLevel.IRREVERSIBLE,
        tags=["compliance", "privacy", "irreversible"],
    )
    async def tool_delete_customer(customer_id: str, data_scope: str) -> dict[str, Any]:
        return await delete_customer_data(
            customer_id=customer_id, data_scope=data_scope, account_store=accounts
        )

    # Register all 16 tools
    for t in (
        tool_search_kb,
        tool_search_web,
        tool_get_longterm,
        tool_get_episodic,
        tool_ticket_status,
        tool_lookup_customer,
        tool_save_longterm,
        tool_update_ticket,
        tool_escalate_tier2,
        tool_delete_lt_all,
        tool_delete_lt_specific,
        tool_delete_ep_all,
        tool_delete_ep_episodes,
        tool_purge_expired,
        tool_execute_refund,
        tool_delete_customer,
    ):
        registry.register(t)

    return registry
