"""Irreversible, high-risk operational tools requiring human-in-the-loop approval."""

from __future__ import annotations

import uuid
from typing import Any

from genai_core.models import ActionStatus

from ..memory.manager import AgentMemoryManager
from ..models import CustomerAccount


async def delete_longterm_memory_all(
    user_id: str,
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Permanently delete all long-term memory records belonging to the authorized user."""
    if not memory_manager:
        return {"error": "Memory manager is not configured."}
    return await memory_manager.delete_longterm_all(user_id=user_id)


async def delete_longterm_memory_specific(
    user_id: str,
    record_id: str,
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Permanently delete a specific long-term memory record by its record ID."""
    if not memory_manager:
        return {"error": "Memory manager is not configured."}
    return await memory_manager.delete_longterm_specific(user_id=user_id, record_id=record_id)


async def delete_episodic_memory_all(
    user_id: str,
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Permanently delete all past episodic action logs belonging to the authorized user."""
    if not memory_manager:
        return {"error": "Memory manager is not configured."}
    return await memory_manager.delete_episodic_all(user_id=user_id)


async def delete_episodic_memory_episodes(
    user_id: str,
    episode_ids: list[str],
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Permanently delete selected episodic action records by ID for the authorized user."""
    if not memory_manager:
        return {"error": "Memory manager is not configured."}
    return await memory_manager.delete_episodic_specific(user_id=user_id, episode_ids=episode_ids)


async def purge_expired(
    user_id: str | None = None,
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Purge expired memory records. Never deletes active unexpired records."""
    if not memory_manager:
        return {"error": "Memory manager is not configured."}
    return await memory_manager.purge_expired_records(user_id=user_id)


async def execute_refund(
    customer_id: str,
    amount_usd: float,
    reason: str,
    account_store: dict[str, CustomerAccount] | None = None,
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Issue a financial refund to the customer's balance. Gated by support-team approval."""
    if account_store is None:
        return {"error": "Account store is not configured."}

    account = account_store.get(customer_id)
    if not account:
        return {"error": f"Customer account '{customer_id}' not found."}

    if amount_usd <= 0.0:
        return {"error": "Refund amount must be strictly greater than $0.00."}

    args_payload = {"customer_id": customer_id, "amount_usd": round(amount_usd, 2)}

    # Episodic idempotency guard to prevent duplicate refunds
    if memory_manager is not None:
        check = await memory_manager.check_action_idempotency(
            user_id=customer_id,
            action_type="execute_refund",
            parameters=args_payload,
        )
        if check.status == "do_not_repeat":
            return {
                "error": f"Duplicate action detected: An identical refund was already processed: {check.reason}",
                "status": "duplicate_blocked",
            }

    # Execute refund
    tx_id = f"tx_rf_{uuid.uuid4().hex[:8]}"
    account.balance_usd += amount_usd

    result = {
        "transaction_id": tx_id,
        "customer_id": customer_id,
        "refund_amount_usd": amount_usd,
        "new_balance_usd": account.balance_usd,
        "reason": reason,
        "status": "completed",
    }

    # Record action episode in memory
    if memory_manager is not None:
        await memory_manager.record_action_outcome(
            user_id=customer_id,
            action_type="execute_refund",
            parameters=args_payload,
            result_summary=f"Refund of ${amount_usd:.2f} executed.",
            status=ActionStatus.SUCCESS,
        )

    return result


async def delete_customer_data(
    customer_id: str,
    data_scope: str,
    account_store: dict[str, CustomerAccount] | None = None,
) -> dict[str, Any]:
    """Permanently delete customer personal data within authorized scope (e.g., GDPR erasure)."""
    if account_store is None:
        return {"error": "Account store is not configured."}

    account = account_store.get(customer_id)
    if not account:
        return {"error": f"Customer account '{customer_id}' not found."}

    normalized_scope = data_scope.strip().lower()
    if normalized_scope not in ("profile", "all", "ticket_history"):
        return {
            "error": f"Invalid data_scope '{data_scope}'. Allowed: 'profile', 'ticket_history', 'all'."
        }

    if normalized_scope == "all":
        del account_store[customer_id]
        details = f"Account '{customer_id}' and all associated records permanently purged."
    elif normalized_scope == "ticket_history":
        account.active_tickets.clear()
        details = f"Active tickets cleared for customer '{customer_id}'."
    else:
        account.name = "[REDACTED]"
        account.email = f"redacted_{customer_id}@deleted.local"
        details = f"Profile personal data scrubbed for customer '{customer_id}'."

    return {
        "customer_id": customer_id,
        "data_scope": normalized_scope,
        "status": "data_deleted",
        "details": details,
        "irreversible": True,
    }
