from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Response model representing generated vector embeddings and metadata."""

    embeddings: list[list[float]] = Field(
        ...,
        description="List of float vectors, strictly preserving the original input order.",
    )
    model: str = Field(
        ...,
        description="The exact embedding model used to produce the vectors.",
    )
    provider: str = Field(
        ...,
        description="The embedding provider (e.g. 'local', 'google', 'openai').",
    )
    dimension: int = Field(
        ...,
        description="The length / dimension of each embedding vector.",
    )
    usage: dict[str, int] | None = Field(
        default=None,
        description="Optional token usage metadata. None when token metrics are not available.",
    )
    latency_ms: float = Field(
        ...,
        description="Total processing latency in milliseconds.",
    )

    @property
    def dimensions(self) -> int:
        """Alias for dimension."""
        return self.dimension

    model_config = {
        "extra": "ignore",
    }
