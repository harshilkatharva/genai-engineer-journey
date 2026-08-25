from time import perf_counter
from uuid import UUID

from rag_app.db.retrive_db import RetriveDBManager
from rag_app.models import RetriveResult
from rag_app.observability.logger import logger
from rag_app.utils.unique_results import fetch_unique_results


class KeywordSearch:
    def __init__(self):
        self.retrive_db_manager = RetriveDBManager()

    async def retrive(
        self,
        tenant_id: UUID,
        queries: list[str],
        top_k_candidates: int,
    ) -> list[RetriveResult]:
        retrieval_start_time = perf_counter()

        results: list[RetriveResult] = []

        for query in queries:
            result = await self.retrive_db_manager.retrive_keyword_chunks(
                tenant_id=tenant_id,
                query=query,
                top_k=top_k_candidates,
            )

            results.extend(result)

        retrieval_latency_ms = (perf_counter() - retrieval_start_time) * 1000

        final_results = fetch_unique_results(results=results, top_k_candidates=top_k_candidates)

        logger.info(
            "Candidate Chunks retrieved successfully using Keyword",
            event="candidate_retrival_completed",
            component="keyword_search",
            latency_ms=retrieval_latency_ms,
            technique="keyword_search",
            top_k=top_k_candidates,
            no_of_chunks=len(results),
        )

        return final_results
