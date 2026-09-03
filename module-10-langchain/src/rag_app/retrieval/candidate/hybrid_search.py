import asyncio
from time import perf_counter
from uuid import UUID

from rag_app.core.settings import get_settings
from rag_app.models import RetriveResult
from rag_app.observability.logger import logger

from .keyword_search import KeywordSearch
from .vector_search import VectorSearch


class HybridSearch:
    def __init__(self):
        self.settings = get_settings()
        self.keyword_search = KeywordSearch()
        self.vector_search = VectorSearch()

    async def retrive(
        self,
        tenant_id: UUID,
        queries: list[str],
        top_k_candidates: int,
    ) -> list[RetriveResult]:
        retrieval_start_time = perf_counter()

        vector_results, keyword_results = await asyncio.gather(
            self._retrive_vectors(
                tenant_id=tenant_id,
                queries=queries,
                top_k_candidates=top_k_candidates,
            ),
            self._retrive_keyword(
                tenant_id=tenant_id,
                queries=queries,
                top_k_candidates=top_k_candidates,
            ),
        )

        normalized_vector_results = self._normalize_scores(vector_results)
        normalized_keyword_results = self._normalize_scores(keyword_results)

        hybrid_results = self._merge_results(
            vector_results=normalized_vector_results,
            keyword_results=normalized_keyword_results,
        )

        final_results = sorted(
            hybrid_results.values(),
            key=lambda result: result.similarity_score,
            reverse=True,
        )[:top_k_candidates]

        retrieval_latency_ms = (perf_counter() - retrieval_start_time) * 1000

        logger.info(
            "Hybrid candidate chunks retrieved successfully",
            event="candidate_retrival_completed",
            component="hybrid_search",
            latency_ms=retrieval_latency_ms,
            technique="hybrid_search",
            top_k=top_k_candidates,
            vector_weight=self.settings.default_vector_search_weight,
            keyword_weight=self.settings.default_keyword_search_weight,
            vector_results_count=len(vector_results),
            keyword_results_count=len(keyword_results),
            final_results_count=len(final_results),
        )

        return final_results

    async def _retrive_vectors(
        self,
        tenant_id: UUID,
        queries: list[str],
        top_k_candidates: int,
    ):
        return await self.vector_search.retrive(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidates=top_k_candidates,
        )

    async def _retrive_keyword(
        self,
        tenant_id: UUID,
        queries: list[str],
        top_k_candidates: int,
    ):
        return await self.keyword_search.retrive(
            tenant_id=tenant_id,
            queries=queries,
            top_k_candidates=top_k_candidates,
        )

    def _normalize_scores(
        self,
        results: list[RetriveResult],
    ) -> list[RetriveResult]:
        if not results:
            return []

        scores = [result.similarity_score for result in results]

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            normalized_score = 1.0

            return [
                RetriveResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    similarity_score=normalized_score,
                )
                for result in results
            ]

        normalized_results = []

        for result in results:
            normalized_score = (result.similarity_score - min_score) / (max_score - min_score)

            normalized_results.append(
                RetriveResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    similarity_score=normalized_score,
                )
            )

        return normalized_results

    def _merge_results(
        self,
        vector_results: list[RetriveResult],
        keyword_results: list[RetriveResult],
    ) -> dict[str, RetriveResult]:
        vector_weight = self.settings.default_vector_search_weight

        keyword_weight = self.settings.default_keyword_search_weight

        if not vector_weight >= 0 and keyword_weight >= 0 and vector_weight + keyword_weight == 1:
            raise ValueError("Vector and Keyword weight not workable fix it.")

        merged_results: dict[str, RetriveResult] = {}

        for result in vector_results:
            hybrid_score = result.similarity_score * vector_weight

            merged_results[result.chunk_id] = RetriveResult(
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                similarity_score=hybrid_score,
            )

        for result in keyword_results:
            keyword_score = result.similarity_score * keyword_weight

            existing = merged_results.get(result.chunk_id)

            if existing is None:
                merged_results[result.chunk_id] = RetriveResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    similarity_score=keyword_score,
                )
            else:
                existing.similarity_score += keyword_score

        return merged_results
