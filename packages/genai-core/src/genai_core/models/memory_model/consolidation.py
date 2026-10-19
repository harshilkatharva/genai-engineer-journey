from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .long_term_record import LongTermMemoryRecord, MemoryCandidate


class ConsolidationStatus(StrEnum):
    """Status classification for memory consolidation outcomes."""

    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"
    FAILED = "failed"


class ConsolidatedMemoryItem(BaseModel):
    """Detailed result for an individual memory candidate evaluated during consolidation."""

    status: ConsolidationStatus = Field(
        ...,
        description="Consolidation outcome: created, updated, skipped, or failed.",
    )
    record: LongTermMemoryRecord | None = Field(
        default=None,
        description="The persisted long-term memory record if created or updated.",
    )
    candidate: MemoryCandidate | None = Field(
        default=None,
        description="The evaluated candidate fact or preference.",
    )
    reason: str | None = Field(
        default=None,
        description="Explanation when a candidate is skipped, updated, or failed.",
    )
    superseded_ids: list[str] = Field(
        default_factory=list,
        description="IDs of previous records superseded by this entry.",
    )

    model_config = {"extra": "ignore"}


class ConsolidationResult(BaseModel):
    """Structured response detailing the outcome of a knowledge consolidation operation."""

    user_id: str = Field(
        ...,
        description="Target user ID for the consolidation operation.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID if conversation context was retrieved from short-term memory.",
    )
    created: list[LongTermMemoryRecord] = Field(
        default_factory=list,
        description="Newly established long-term memory records.",
    )
    updated: list[LongTermMemoryRecord] = Field(
        default_factory=list,
        description="Existing memory records that were updated or superseded.",
    )
    skipped: list[ConsolidatedMemoryItem] = Field(
        default_factory=list,
        description="Candidates skipped due to duplication, redundancy, or sensitivity rules.",
    )
    failed: list[ConsolidatedMemoryItem] = Field(
        default_factory=list,
        description="Candidates that failed validation, embedding, or database persistence.",
    )
    total_extracted: int = Field(
        default=0,
        description="Total memory candidates extracted from the conversation by the LLM.",
    )
    total_processed: int = Field(
        default=0,
        description="Total items evaluated and processed during consolidation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Operational metadata such as provider used, model, or latency.",
    )

    model_config = {"extra": "ignore"}


class MemoryExtractionOutput(BaseModel):
    """Structured extraction payload schema for LLM structured output completion."""

    memories: list[MemoryCandidate] = Field(
        default_factory=list,
        description="List of memory candidates extracted from conversation context.",
    )

    model_config = {"extra": "ignore"}
