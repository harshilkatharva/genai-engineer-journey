from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models.retrieval_model import DocumentChunk, RetrievalResult


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """Protocol defining the standard interface for vector storage and search backends."""

    @property
    def dimensions(self) -> int:
        """The vector dimension expected by this store or collection."""
        ...

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Search the vector store using an embedding vector.

        Returns matching chunks ordered descending by relevance score.
        """
        ...

    async def upsert(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """Add or update document chunks in the store.

        All chunks passed to this method must have their embedding vectors populated.
        Returns the number of chunks successfully stored.
        """
        ...

    async def delete(
        self,
        chunk_ids: list[str],
    ) -> int:
        """Delete document chunks by ID.

        Returns the number of chunks deleted.
        """
        ...
