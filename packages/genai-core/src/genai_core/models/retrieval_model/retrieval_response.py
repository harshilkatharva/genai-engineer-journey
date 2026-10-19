from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Represents a single retrieved document chunk with relevance score and metadata."""

    id: str = Field(
        ...,
        description="Unique identifier of the retrieved chunk.",
    )
    content: str = Field(
        ...,
        description="Text content of the retrieved chunk.",
    )
    score: float = Field(
        ...,
        description="Relevance or similarity score (typically normalized between 0.0 and 1.0).",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-indexed rank of this result within the returned response.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved arbitrary metadata associated with the chunk.",
    )
    source_id: str | None = Field(
        default=None,
        description="Identifier of the original source document.",
    )

    model_config = {
        "extra": "ignore",
    }


class RetrievalResponse(BaseModel):
    """Response model containing retrieved results, count, and execution metrics."""

    results: list[RetrievalResult] = Field(
        default_factory=list,
        description="List of retrieved results, ordered descending by relevance score.",
    )
    total_results: int = Field(
        ...,
        description="Number of results returned.",
    )
    latency_ms: float = Field(
        ...,
        description="Total retrieval latency in milliseconds.",
    )
    query: str | None = Field(
        default=None,
        description="The original natural language query string if provided in the request.",
    )
    search_mode: str = Field(
        default="dense",
        description="The search mode utilized to produce these results.",
    )
    reranked: bool = Field(
        default=False,
        description="Indicates whether the results were re-ranked using a cross-encoder.",
    )
    rerank_message: str | None = Field(
        default=None,
        description="Optional status message if re-ranking was requested but skipped or had warnings.",
    )

    model_config = {
        "extra": "ignore",
    }
