from uuid import UUID

# from time import perf_counter
from pgvector import Vector

from rag_app.db.retrive_db import RetriveDBManager
from rag_app.embedding.embedding_manager import (
    EmbeddingManager,
)

from rag_app.models import RetriveResult


class VectorSearch:
    def __init__(self):
        self.retrive_db_manager = RetriveDBManager()
        self.embedding_manager = EmbeddingManager()

    async def retrive(self, tenant_id: UUID, queries: list[str], top_k_candidates: int):
        # total_start_time = perf_counter()

        # embedding_start_time = perf_counter()
        query_embeddings = [self.embedding_manager.embed_query(query) for query in queries]
        # embedding_latency_ms = (perf_counter() - embedding_start_time) * 1000

        # retrieval_start_time = perf_counter()
        results: list[RetriveResult] = []

        for query_embedding in query_embeddings:
            result = await self.retrive_db_manager.retrive_chunks(
                tenant_id, Vector(query_embedding), top_k_candidates
            )
            for val in result:
                results.append(
                    RetriveResult(chunk_text=val.chunk_text, similarity_score=val.similarity_score)
                )

        # retrieval_latency_ms = (perf_counter() - retrieval_start_time) * 1000

        # total_latency_ms = (perf_counter() - total_start_time) * 1000

        # # Log retrieval metrics
        # retrive_tracker = RetriveTracker(
        #     tenant_id=str(tenant_id),
        #     query=queries,
        #     # top_k=top_k or self.settings.default_top_k,
        #     results_count=len(results),
        #     retrieval_latency_ms=retrieval_latency_ms,
        #     embedding_latency_ms=embedding_latency_ms,
        #     total_latency_ms=total_latency_ms,
        # )
        # self.retrive_tracker_logger.track(retrive_tracker)

        return results
