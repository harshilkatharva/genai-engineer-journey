from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ActionStatus(StrEnum):
    """Execution status for an action episode."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


class ActionEpisode(BaseModel):
    """An episodic record of a completed or attempted tool action/event."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the action episode.",
    )
    user_id: str = Field(
        ...,
        description="User identifier owning the episode.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session or conversation ID where the action occurred.",
    )
    action_type: str = Field(
        ...,
        description="Type or name of the action (e.g. tool name, API call).",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters passed to the action.",
    )
    parameter_hash: str = Field(
        ...,
        description="Deterministic canonical SHA-256 hash of the parameters.",
    )
    result_summary: str | None = Field(
        default=None,
        description="Concise summary or return value snippet of the action result.",
    )
    status: ActionStatus | str = Field(
        default=ActionStatus.SUCCESS,
        description="Execution status of the episode.",
    )
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the action occurred.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional episode metadata.",
    )

    model_config = {"extra": "ignore"}


class ActionCheck(BaseModel):
    """Decision check for idempotency and loop prevention."""

    status: Literal["execute", "do_not_repeat"] = Field(
        ...,
        description="Whether to proceed with execution or skip duplicate action.",
    )
    reason: str = Field(
        ...,
        description="Explanation for the execution decision.",
    )
    previous_episode: ActionEpisode | None = Field(
        default=None,
        description="Reference to previous matching episode if status is 'do_not_repeat'.",
    )

    model_config = {"extra": "ignore"}
