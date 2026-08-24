from __future__ import annotations
from functools import lru_cache

import asyncio
from time import perf_counter
from uuid import UUID

from sentence_transformers import SentenceTransformer

from rag_app.core import get_settings
from rag_app.tracker.embedding_tracker import (
    EmbeddingTrackerLogger,
)
from rag_app.models.chunk.chunk import Chunk
from rag_app.models.tracker.embedding_tracker import (
    EmbeddingTracker,
)
from rag_app.user_data.data_manager import DataManager


class EmbeddingManager:
    """
    Creates local embeddings for chunks using sentence-transformers.

    The model is loaded once per manager instance and reused for all
    embedding operations.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.data_manager = DataManager()
        self.tracker_logger = EmbeddingTrackerLogger()
        self.model = get_embedding_model()

    async def embed_chunks(
        self,
        tenant_id: UUID,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """
        Embed a list of chunks in batches.

        Returns:
            One embedding vector per input chunk.
        """
        if not chunks:
            return []

        started_at = perf_counter()

        texts = [chunk.text for chunk in chunks]

        total_tokens = sum(chunk.token_count for chunk in chunks)

        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedding_vectors = [vector.tolist() for vector in embeddings]

        latency_ms = (perf_counter() - started_at) * 1000

        estimated_cost = self._calculate_cost(total_tokens)

        tracker = EmbeddingTracker(
            tenant_id=str(tenant_id),
            embedding_model=self.settings.embedding_model,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
        )

        self.tracker_logger.track(tracker)

        return embedding_vectors

    async def embed_documents(
        self,
        tenant_id: UUID,
        documents: dict[str, list[Chunk]],
    ) -> dict[str, list[list[float]]]:
        """
        Embed chunks grouped by document.

        Returns:
            {
                "document_id": [[embedding], [embedding], ...]
            }
        """
        result: dict[str, list[list[float]]] = {}

        for document_id, chunks in documents.items():
            result[document_id] = await self.embed_chunks(
                tenant_id=tenant_id,
                chunks=chunks,
            )

        return result

    async def embed_query(
        self,
        query: str,
    ) -> tuple[list[float], int, int]:
        """
        Create a normalized embedding for a search query.

        Query embeddings are not persisted because they are transient
        retrieval inputs.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = await asyncio.to_thread(
            self.model.encode,
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        token_counts = 0
        estimated_cost = 0

        return (embedding.tolist(), token_counts, estimated_cost)

    def _calculate_cost(
        self,
        total_tokens: int,
    ) -> float:
        return total_tokens / 1_000_000 * self.settings.embedding_cost_per_million_tokens


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()

    return SentenceTransformer(settings.default_embedding_model)
