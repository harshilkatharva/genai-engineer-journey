from uuid import UUID

from time import perf_counter
from pgvector import Vector

from rag_app.db.retrive_db import RetriveDBManager
from rag_app.embedding.embedding_manager import (
    EmbeddingManager,
)
from rag_app.models import RetriveResult

from rag_app.observability.logger import logger


class VectorSearch:
    def __init__(self):
        self.retrive_db_manager = RetriveDBManager()
        self.embedding_manager = EmbeddingManager()

    async def retrive(self, tenant_id: UUID, queries: list[str], top_k_candidates: int):
        embedding_start_time = perf_counter()
        query_embeddings = []
        token_count = 0
        cost_usd = 0

        for query in queries:
            embedding, tokens, cost = await self.embedding_manager.embed_query(query)

            query_embeddings.append(embedding)
            token_count += tokens
            cost_usd += cost

        embedding_latency_ms = (perf_counter() - embedding_start_time) * 1000

        logger.info(
            "Query embeddings generated",
            event="query_embedding_completed",
            component="embedding",
            latency_ms=embedding_latency_ms,
            token_count=token_count,
            cost_usd=cost_usd,
        )

        retrieval_start_time = perf_counter()
        results: list[RetriveResult] = []

        for query_embedding in query_embeddings:
            result = await self.retrive_db_manager.retrive_chunks(
                tenant_id, Vector(query_embedding), top_k_candidates
            )
            for val in result:
                results.append(
                    RetriveResult(
                        chunk_id=val.chunk_id,
                        chunk_text=val.chunk_text,
                        similarity_score=val.similarity_score,
                    )
                )

        retrieval_latency_ms = (perf_counter() - retrieval_start_time) * 1000

        unique_results: dict[str, RetriveResult] = {}

        for result in results:
            existing = unique_results.get(result.chunk_id)

            if existing is None or result.similarity_score > existing.similarity_score:
                unique_results[result.chunk_id] = result

        # Sort by highest similarity score first
        final_results = sorted(
            unique_results.values(),
            key=lambda x: x.similarity_score,
            reverse=True,
        )[:top_k_candidates]

        logger.info(
            "Candidate Chunks retrieved successfully",
            event="candidate_retrival_completed",
            component="vector_search",
            latency_ms=retrieval_latency_ms,
            technique="vector_search",
            top_k=top_k_candidates,
            no_of_chunks=len(results),
        )

        return final_results
