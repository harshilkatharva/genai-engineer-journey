from __future__ import annotations

import math
from typing import Any

from ..exceptions import IncompatibleEmbeddingError
from ..models.retrieval_model import DocumentChunk, RetrievalResult


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    val = dot / (norm_a * norm_b)
    # Clamp to valid [-1.0, 1.0] range
    return max(-1.0, min(1.0, val))


class InMemoryVectorStore:
    """In-memory reference vector store supporting exact cosine similarity search.

    Provides a zero-dependency vector store suitable for local development, unit testing,
    and fast in-process retrieval.
    """

    def __init__(
        self,
        dimensions: int,
        chunks: list[DocumentChunk] | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be greater than 0, got {dimensions}.")
        self._dimensions = dimensions
        self._chunks: dict[str, DocumentChunk] = {}

        if chunks:
            self._upsert_sync(chunks)

    @property
    def dimensions(self) -> int:
        """The vector dimension required by this store."""
        return self._dimensions

    def _upsert_sync(self, chunks: list[DocumentChunk]) -> int:
        count = 0
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.id}' cannot be indexed because its embedding vector is None."
                )
            if len(chunk.embedding) != self._dimensions:
                raise IncompatibleEmbeddingError(
                    requested=len(chunk.embedding),
                    supported=self._dimensions,
                    message=(
                        f"Chunk '{chunk.id}' embedding dimension ({len(chunk.embedding)}) "
                        f"does not match store dimension ({self._dimensions})."
                    ),
                )
            self._chunks[chunk.id] = chunk
            count += 1
        return count

    async def upsert(self, chunks: list[DocumentChunk]) -> int:
        """Add or update document chunks in the in-memory store."""
        return self._upsert_sync(chunks)

    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete document chunks by ID."""
        deleted = 0
        for cid in chunk_ids:
            if cid in self._chunks:
                del self._chunks[cid]
                deleted += 1
        return deleted

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Search in-memory chunks using cosine similarity."""
        if len(query_vector) != self._dimensions:
            raise IncompatibleEmbeddingError(
                requested=len(query_vector),
                supported=self._dimensions,
                message=(
                    f"Query vector dimension ({len(query_vector)}) does not match "
                    f"store dimension ({self._dimensions})."
                ),
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}.")

        scored_candidates: list[tuple[DocumentChunk, float]] = []

        for chunk in self._chunks.values():
            if chunk.embedding is None:
                continue

            # Apply metadata filters if provided
            if filters:
                match = True
                for key, expected_value in filters.items():
                    actual_val = chunk.metadata.get(key)
                    if actual_val != expected_value:
                        match = False
                        break
                if not match:
                    continue

            similarity = _cosine_similarity(query_vector, chunk.embedding)

            if score_threshold is not None and similarity < score_threshold:
                continue

            scored_candidates.append((chunk, similarity))

        # Sort descending by score
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(scored_candidates[:top_k], 1):
            results.append(
                RetrievalResult(
                    id=chunk.id,
                    content=chunk.content,
                    score=round(score, 6),
                    rank=rank,
                    metadata=chunk.metadata,
                    source_id=chunk.source_id,
                )
            )

        return results

    async def search_keyword(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Search in-memory chunks using keyword token overlap matching."""
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}.")

        import re

        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return []

        scored_candidates: list[tuple[DocumentChunk, float]] = []

        for chunk in self._chunks.values():
            if filters:
                match = True
                for key, expected_value in filters.items():
                    actual_val = chunk.metadata.get(key)
                    if actual_val != expected_value:
                        match = False
                        break
                if not match:
                    continue

            chunk_tokens = re.findall(r"\w+", chunk.content.lower())
            if not chunk_tokens:
                continue

            chunk_tokens_set = set(chunk_tokens)
            overlap = query_tokens.intersection(chunk_tokens_set)
            if not overlap:
                continue

            overlap_ratio = len(overlap) / len(query_tokens)
            # Frequency factor
            freq_count = sum(chunk_tokens.count(t) for t in overlap)
            freq_score = min(1.0, freq_count / max(1, len(chunk_tokens)))
            score = (overlap_ratio * 0.7) + (freq_score * 0.3)

            if score_threshold is not None and score < score_threshold:
                continue

            scored_candidates.append((chunk, score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(scored_candidates[:top_k], 1):
            results.append(
                RetrievalResult(
                    id=chunk.id,
                    content=chunk.content,
                    score=round(score, 6),
                    rank=rank,
                    metadata=chunk.metadata,
                    source_id=chunk.source_id,
                )
            )

        return results
