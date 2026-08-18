from __future__ import annotations

from time import perf_counter
from uuid import UUID

from pgvector import Vector

from semantic_search_eng.config import get_settings
from semantic_search_eng.db.retrive_db import RetriveDBManager
from semantic_search_eng.embedding.embedding_manager import (
    EmbeddingManager,
)
from semantic_search_eng.logger.retrive_tracker import (
    RetriveTrackerLogger,
)
from semantic_search_eng.models.retrive_tracker import RetriveTracker


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
        self.retrive_tracker_logger = RetriveTrackerLogger()
        self.retrive_db_manager = RetriveDBManager()
        self.embedding_manager = EmbeddingManager()

    async def retrieve(
        self,
        tenant_id: UUID,
        query: str,
        top_k: int | None = None,
        document_type: str | None = None,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = self.settings.default_top_k

        if top_k > self.settings.max_top_k:
            raise ValueError(f"Can not retrive more than {self.settings.max_top_k}")

        total_start_time = perf_counter()

        embedding_start_time = perf_counter()
        query_embedding = self.embedding_manager.embed_query(query)
        embedding_latency_ms = (perf_counter() - embedding_start_time) * 1000

        retrieval_start_time = perf_counter()
        results = await self.retrive_db_manager.retrive_chunks(
            tenant_id, Vector(query_embedding), top_k, document_type=document_type
        )
        retrieval_latency_ms = (perf_counter() - retrieval_start_time) * 1000

        total_latency_ms = (perf_counter() - total_start_time) * 1000

        # Log retrieval metrics
        retrive_tracker = RetriveTracker(
            tenant_id=str(tenant_id),
            query=query,
            top_k=top_k or self.settings.default_top_k,
            results_count=len(results),
            retrieval_latency_ms=retrieval_latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            total_latency_ms=total_latency_ms,
        )
        self.retrive_tracker_logger.track(retrive_tracker)

        return results
