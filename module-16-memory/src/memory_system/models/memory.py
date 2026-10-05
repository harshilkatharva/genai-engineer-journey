from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

MessageRole = Literal["system", "user", "assistant", "tool"]
MemoryCategory = Literal["fact", "preference", "goal", "constraint", "other"]
ActionStatus = Literal["attempted", "completed", "failed"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversationMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)


class PinnedGoal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1, max_length=4_000)
    created_at: datetime = Field(default_factory=utc_now)


class WorkingContext(BaseModel):
    pinned_goals: list[PinnedGoal]
    messages: list[ConversationMessage]
    token_count: int = Field(ge=0)
    token_budget: int = Field(ge=0)


class MemoryCandidate(BaseModel):
    content: str = Field(min_length=1, max_length=4_000)
    category: MemoryCategory = "fact"
    supersedes_ids: list[UUID] = Field(default_factory=list)
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def validate_expiration_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        return value


class LongTermMemoryRecord(MemoryCandidate):
    id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=255)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def validate_creation_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class MemorySearchResult(BaseModel):
    memory: LongTermMemoryRecord
    similarity: float
    rank_score: float
    superseded: bool = False


class ActionEpisode(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=255)
    task_id: str = Field(min_length=1, max_length=255)
    conversation_id: str | None = Field(default=None, max_length=255)
    action_type: str = Field(min_length=1, max_length=255)
    target: str = Field(min_length=1, max_length=1_000)
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = "attempted"
    occurred_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    @field_validator("occurred_at", "expires_at")
    @classmethod
    def validate_timestamp_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("episode timestamps must include a timezone")
        return value

    model_config = ConfigDict(extra="forbid")


class ActionCheck(BaseModel):
    already_attempted: bool
    recommendation: Literal["execute", "do_not_repeat"]
    episode: ActionEpisode | None = None
