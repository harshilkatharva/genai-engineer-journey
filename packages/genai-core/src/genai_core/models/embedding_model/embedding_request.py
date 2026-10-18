from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    input: str | list[str] = Field(
        ...,
        description="Text string or list of text strings to embed.",
    )
    batch_size: int | None = Field(
        default=None,
        description="Optional request-level batch size overriding the service default.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to return L2-normalized unit vectors suitable for cosine similarity.",
    )

    model_config = {
        "extra": "ignore",
    }
