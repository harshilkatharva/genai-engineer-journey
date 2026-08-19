from __future__ import annotations
import asyncio
from time import perf_counter
from uuid import UUID

from sentence_transformers import SentenceTransformer

from rag_app.config import get_settings
from rag_app.logger.embedding_tracker import (
    EmbeddingTrackerLogger,
)
from rag_app.models.chunk import Chunk
from rag_app.models.embedding_tracker import (
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

        self.model = SentenceTransformer(self.settings.embedding_model)

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
        save: bool = True,
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

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Create a normalized embedding for a search query.

        Query embeddings are not persisted because they are transient
        retrieval inputs.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def _save_embeddings(
        self,
        tenant_id: UUID,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store embeddings grouped by document.

        Each document gets one embedding file containing vectors in the
        same chunk order as the corresponding chunk file.
        """
        grouped: dict[str, list[list[float]]] = {}

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            grouped.setdefault(
                chunk.document_id,
                [],
            ).append(embedding)

        for document_id, document_embeddings in grouped.items():
            self.data_manager.save_embeddings(
                tenant_id=tenant_id,
                document_id=document_id,
                embeddings=document_embeddings,
            )

    def _calculate_cost(
        self,
        total_tokens: int,
    ) -> float:
        return total_tokens / 1_000_000 * self.settings.embedding_cost_per_million_tokens
