"""Domain models for customer accounts, support tickets, and undo tracking."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TicketStatus(StrEnum):
    """Lifecycle status of a support ticket."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    ESCALATED_TIER2 = "escalated_tier2"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CustomerAccount(BaseModel):
    """Customer account record stored in operational datastore."""

    customer_id: str
    name: str
    email: str
    tier: str = "standard"  # standard, gold, vip
    balance_usd: float = 0.0
    active_tickets: list[str] = Field(default_factory=list)
    risk_flag: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportTicket(BaseModel):
    """Support ticket tracked in operational datastore."""

    ticket_id: str
    customer_id: str
    title: str
    description: str
    status: TicketStatus = TicketStatus.OPEN
    assigned_team: str = "tier1_support"
    escalation_reason: str | None = None
    priority: str = "medium"  # low, medium, high, urgent
    history: list[dict[str, Any]] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ReversalAuditRecord(BaseModel):
    """Tracks state modifications to allow explicit reversibility."""

    operation_id: str
    tool_name: str
    target_entity: str
    target_id: str
    previous_state: dict[str, Any]
    new_state: dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
    reverted: bool = False
