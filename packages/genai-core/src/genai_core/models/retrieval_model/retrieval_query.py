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
        description="Search mode: 'dense' (vector search), 'sparse' (full-text keyword search), or 'hybrid' (combined dense + sparse).",
    )
    vector_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional weight for vector scores in hybrid search (default 0.6).",
    )
    keyword_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional weight for keyword scores in hybrid search (default 0.4).",
    )
    rerank: bool | None = Field(
        default=None,
        description="Whether to execute cross-encoder re-ranking on candidate results.",
    )
    rerank_top_k: int | None = Field(
        default=None,
        gt=0,
        description="Number of final results to retain after re-ranking. Defaults to top_k.",
    )
    on_rerank_unavailable: Literal["raise", "warn_and_continue"] = Field(
        default="warn_and_continue",
        description="Behavior when re-ranking is requested but unavailable ('raise' or 'warn_and_continue').",
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

        if self.search_mode in ("sparse", "hybrid") and not has_text:
            raise ValueError(f"Search mode '{self.search_mode}' requires a text 'query' string.")

        if (
            self.vector_weight is not None
            and self.keyword_weight is not None
            and round(self.vector_weight + self.keyword_weight, 4) != 1.0
        ):
            raise ValueError("vector_weight and keyword_weight must sum to 1.0.")

        return self

    model_config = {
        "extra": "ignore",
    }
