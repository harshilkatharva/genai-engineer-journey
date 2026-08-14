from __future__ import annotations

from time import perf_counter

import numpy as np

from semantic_search_eng.config import get_settings
from semantic_search_eng.embedding.embedding_manager import (
    EmbeddingManager,
)
from semantic_search_eng.logger.query_tracker import (
    QueryTrackerLogger,
)
from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.models.query_tracker import QueryTracker
from semantic_search_eng.models.search_response import SearchResult


class RetriverManager:
    """
    Retrieves the most semantically similar chunks for a query.

    Module 7 implementation:
        local JSON storage + brute-force cosine similarity

    Module 8 can replace this implementation with a real VectorStore
    without changing the API contract.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_manager = EmbeddingManager()
        self.query_tracker_logger = QueryTrackerLogger()

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        if not chunks:
            return []

        resolved_top_k = self._resolve_top_k(top_k)

        total_started_at = perf_counter()

        # --------------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------------
        embedding_started_at = perf_counter()

        query_embedding = self.embedding_manager.embed_query(query)

        embedding_latency_ms = (perf_counter() - embedding_started_at) * 1000

        # --------------------------------------------------------------
        # Vector search
        # --------------------------------------------------------------
        retrieval_started_at = perf_counter()

        chunk_matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        similarity_scores = chunk_matrix @ query_vector

        ranked_indexes = np.argsort(similarity_scores)[::-1]

        top_indexes = ranked_indexes[:resolved_top_k]

        results = [
            SearchResult(
                chunk=chunks[index],
                similarity_score=float(similarity_scores[index]),
            )
            for index in top_indexes
        ]

        retrieval_latency_ms = (perf_counter() - retrieval_started_at) * 1000

        total_latency_ms = (perf_counter() - total_started_at) * 1000

        # --------------------------------------------------------------
        # Query tracking
        # --------------------------------------------------------------
        query_token_count = self._estimate_tokens(query)

        estimated_cost = (
            query_token_count / 1_000_000 * self.settings.embedding_cost_per_million_tokens
        )

        tracker = QueryTracker(
            tenant_id=tenant_id,
            query=query,
            embedding_model=self.settings.embedding_model,
            query_token_count=query_token_count,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
            estimated_cost=estimated_cost,
            top_k=resolved_top_k,
        )

        self.query_tracker_logger.track(tracker)

        return results

    def _resolve_top_k(
        self,
        top_k: int | None,
    ) -> int:
        value = top_k if top_k is not None else self.settings.default_top_k

        if value < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        if value > self.settings.max_top_k:
            raise ValueError(f"top_k cannot exceed {self.settings.max_top_k}.")

        return value

    @staticmethod
    def _estimate_tokens(
        text: str,
    ) -> int:
        """
        Lightweight token estimate for query cost tracking.

        Actual embedding happens in sentence-transformers.
        """
        return len(text.split())
