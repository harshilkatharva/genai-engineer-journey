"""Reversible state-modifying operational tools for Module 17."""

from __future__ import annotations

import time
from typing import Any

from ..memory.manager import AgentMemoryManager
from ..models import ReversalAuditRecord, SupportTicket, TicketStatus


async def save_longterm_memory(
    user_id: str,
    content: str,
    category: str = "fact",
    memory_manager: AgentMemoryManager | None = None,
) -> dict[str, Any]:
    """Store or update a user fact or preference in persistent long-term memory."""
    if not memory_manager:
        return {"error": "Memory manager is not configured."}
    return await memory_manager.save_longterm_memory(
        user_id=user_id, content=content, category=category
    )


async def update_ticket_status(
    ticket_id: str,
    new_status: str,
    comment: str = "",
    ticket_store: dict[str, SupportTicket] | None = None,
    audit_log: list[ReversalAuditRecord] | None = None,
) -> dict[str, Any]:
    """Update a support ticket status while recording the previous state for reversibility."""
    if ticket_store is None:
        return {"error": "Ticket store is not configured."}

    ticket = ticket_store.get(ticket_id)
    if not ticket:
        return {"error": f"Ticket '{ticket_id}' not found."}

    valid_statuses = [s.value for s in TicketStatus]
    normalized_status = new_status.strip().lower()
    if normalized_status not in valid_statuses:
        return {"error": f"Invalid status '{new_status}'. Allowed values: {valid_statuses}."}

    previous_status = str(ticket.status)
    ticket.status = TicketStatus(normalized_status)
    ticket.updated_at = time.time()
    ticket.history.append(
        {
            "action": "status_update",
            "from": previous_status,
            "to": normalized_status,
            "comment": comment,
            "timestamp": ticket.updated_at,
        }
    )

    if audit_log is not None:
        audit_log.append(
            ReversalAuditRecord(
                operation_id=f"op_{len(audit_log) + 1}",
                tool_name="update_ticket_status",
                target_entity="ticket",
                target_id=ticket_id,
                previous_state={"status": previous_status},
                new_state={"status": normalized_status, "comment": comment},
            )
        )

    return {
        "ticket_id": ticket_id,
        "previous_status": previous_status,
        "current_status": normalized_status,
        "reversible": True,
        "reversal_note": f"Can be reverted by resetting status to '{previous_status}'.",
        "comment": comment,
    }


async def escalate_to_tier2(
    ticket_id: str,
    reason: str,
    urgency: str = "medium",
    ticket_store: dict[str, SupportTicket] | None = None,
    audit_log: list[ReversalAuditRecord] | None = None,
) -> dict[str, Any]:
    """Escalate a support ticket to Tier-2 engineers with supported de-escalation mechanism."""
    if ticket_store is None:
        return {"error": "Ticket store is not configured."}

    ticket = ticket_store.get(ticket_id)
    if not ticket:
        return {"error": f"Ticket '{ticket_id}' not found."}

    prev_team = ticket.assigned_team
    prev_status = str(ticket.status)

    ticket.assigned_team = "tier2_engineering"
    ticket.status = TicketStatus.ESCALATED_TIER2
    ticket.escalation_reason = reason
    ticket.priority = urgency.lower()
    ticket.updated_at = time.time()
    ticket.history.append(
        {
            "action": "escalate_tier2",
            "reason": reason,
            "urgency": urgency,
            "previous_team": prev_team,
            "timestamp": ticket.updated_at,
        }
    )

    if audit_log is not None:
        audit_log.append(
            ReversalAuditRecord(
                operation_id=f"op_{len(audit_log) + 1}",
                tool_name="escalate_to_tier2",
                target_entity="ticket",
                target_id=ticket_id,
                previous_state={"assigned_team": prev_team, "status": prev_status},
                new_state={"assigned_team": "tier2_engineering", "status": "escalated_tier2"},
            )
        )

    return {
        "ticket_id": ticket_id,
        "assigned_team": "tier2_engineering",
        "status": "escalated_tier2",
        "reason": reason,
        "urgency": urgency,
        "reversible": True,
        "reversal_mechanism": f"Can de-escalate back to '{prev_team}' queue.",
    }
