from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class RetrievalQuery(BaseModel):
    """Request model for retrieving relevant documents or chunks."""

    query: str | None = Field(
        default=None,
        description="Natural language query string to be embedded and searched.",
    )
    query_vector: list[float] | None = Field(
        default=None,
        description="Optional pre-computed query embedding vector.",
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Maximum number of relevant results to return.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity score threshold (0.0 to 1.0).",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata key-value filters (e.g. {'tenant_id': '...', 'doc_type': 'pdf'}).",
    )
    search_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="dense",
        description="Search mode. Dense is currently supported; sparse and hybrid are planned for future stages.",
    )

    @model_validator(mode="after")
    def validate_query_present(self) -> RetrievalQuery:
        if self.query is not None and not self.query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        if self.query_vector is not None and len(self.query_vector) == 0:
            raise ValueError("Query vector cannot be empty.")

        has_text = self.query is not None and bool(self.query.strip())
        has_vector = self.query_vector is not None and len(self.query_vector) > 0

        if not has_text and not has_vector:
            raise ValueError(
                "Either 'query' (non-empty string) or 'query_vector' (non-empty list of floats) must be provided."
            )

        return self

    model_config = {
        "extra": "ignore",
    }
