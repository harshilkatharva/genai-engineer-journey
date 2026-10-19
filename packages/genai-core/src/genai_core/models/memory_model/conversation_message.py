from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Role associated with a conversational message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PinnedGoal(BaseModel):
    """Pinned instruction or persistent agent goal that survives context pruning."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the pinned goal.",
    )
    goal: str = Field(
        ...,
        description="The goal or instruction text to keep pinned.",
    )
    pinned_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the goal was pinned.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the pinned goal.",
    )

    model_config = {"extra": "ignore"}


class ConversationMessage(BaseModel):
    """A single turn or message within a conversation."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the message.",
    )
    role: MessageRole | str = Field(
        ...,
        description="Role of the message sender.",
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the message was sent.",
    )
    token_count: int | None = Field(
        default=None,
        description="Calculated or cached token count for the message.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the message.",
    )

    model_config = {"extra": "ignore"}


class WorkingContext(BaseModel):
    """Working context assembled for an LLM call with token budgeting."""

    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Chronological conversation history within budget.",
    )
    pinned_goals: list[PinnedGoal] = Field(
        default_factory=list,
        description="High-priority pinned goals prepended or preserved.",
    )
    total_tokens: int = Field(
        default=0,
        description="Total token consumption of messages and goals.",
    )
    truncated: bool = Field(
        default=False,
        description="Indicates whether older messages were truncated to respect budget.",
    )

    model_config = {"extra": "ignore"}
