from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an indexable text chunk with metadata and optional embedding."""

    id: str = Field(
        ...,
        description="Unique identifier for the document chunk.",
    )
    content: str = Field(
        ...,
        description="Text content of the chunk.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional embedding vector. Can be pre-computed or generated on upsert.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (e.g. document_type, title, page, tenant_id).",
    )
    source_id: str | None = Field(
        default=None,
        description="Identifier of the source document or file.",
    )
    chunk_index: int | None = Field(
        default=None,
        ge=0,
        description="Sequential index of the chunk within the source document.",
    )

    model_config = {
        "extra": "ignore",
    }
