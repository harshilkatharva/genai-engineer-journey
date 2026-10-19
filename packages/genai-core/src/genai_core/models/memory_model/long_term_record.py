from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryCategory(StrEnum):
    """Classification category for long-term memory records."""

    FACT = "fact"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    CONTEXT = "context"


class LongTermMemoryRecord(BaseModel):
    """Persistent memory unit representing facts, preferences, or profile data."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the long-term memory entry.",
    )
    user_id: str = Field(
        ...,
        description="Identifier of the user owning this memory (strict isolation).",
    )
    category: MemoryCategory | str = Field(
        default=MemoryCategory.FACT,
        description="Category classification for the memory.",
    )
    content: str = Field(
        ...,
        description="Extracted or declared factual content.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in the accuracy of the memory (0.0 to 1.0).",
    )
    supersedes_ids: list[str] = Field(
        default_factory=list,
        description="IDs of previous memory records superseded by this record.",
    )
    is_superseded: bool = Field(
        default=False,
        description="Whether this memory record has been superseded by a newer entry.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC creation timestamp.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC last update timestamp.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiration timestamp. Record considered inactive after this time.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to the memory.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector embedding of content for semantic search.",
    )

    model_config = {"extra": "ignore"}


class MemoryCandidate(BaseModel):
    """Candidate fact or preference to evaluate, store, or update in long-term memory."""

    category: MemoryCategory | str = Field(
        default=MemoryCategory.FACT,
        description="Category classification for the candidate.",
    )
    content: str = Field(
        ...,
        description="Text content to be stored.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this candidate.",
    )
    supersedes_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of record IDs this candidate supersedes.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiration timestamp.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the candidate.",
    )

    model_config = {"extra": "ignore"}


class MemorySearchResult(BaseModel):
    """Ranked memory search result incorporating similarity and recency decay."""

    record: LongTermMemoryRecord = Field(
        ...,
        description="The retrieved long-term memory record.",
    )
    similarity: float = Field(
        ...,
        description="Semantic cosine similarity score between query and record content.",
    )
    recency_score: float = Field(
        ...,
        description="Exponential recency score (decayed based on half-life).",
    )
    rank_score: float = Field(
        ...,
        description="Combined rank score: 0.65 * sim + 0.35 * recency - (2.0 if superseded).",
    )

    model_config = {"extra": "ignore"}
