from __future__ import annotations

import asyncio
from functools import lru_cache
from time import perf_counter

from sentence_transformers import CrossEncoder

from rag_app.core import get_settings
from rag_app.models import RetriveResult
from rag_app.observability.logger import logger


class CrossEncoderReranker:
    def __init__(self):
        self.settings = get_settings()

        self.model = get_model()

    async def rerank(
        self,
        queries: list[str],
        results: list[RetriveResult],
        top_k: int,
    ) -> list[RetriveResult]:
        if not results:
            return []

        rerank_start_time = perf_counter()

        pairs = []

        for query in queries:
            for result in results:
                pairs.append(
                    (
                        query,
                        result.chunk_text,
                    )
                )

        scores = await asyncio.to_thread(
            self.model.predict,
            pairs,
            batch_size=self.settings.re_ranker_batch_size,
            show_progress_bar=False,
        )

        # One chunk can be scored against multiple expanded queries.
        # Keep the highest CrossEncoder score for each chunk.
        chunk_scores: dict[str, float] = {}

        for index, result in enumerate(results):
            query_scores = []

            for query_index in range(len(queries)):
                score_index = query_index * len(results) + index

                query_scores.append(float(scores[score_index]))

            chunk_scores[result.chunk_id] = max(query_scores)

        reranked_results = []

        for result in results:
            reranked_results.append(
                RetriveResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    similarity_score=chunk_scores[result.chunk_id],
                )
            )

        reranked_results.sort(
            key=lambda result: result.similarity_score,
            reverse=True,
        )

        final_results = reranked_results[:top_k]

        rerank_latency_ms = (perf_counter() - rerank_start_time) * 1000

        logger.info(
            "Candidate chunks reranked successfully",
            event="reranker_completed",
            component="cross_encoder_reranker",
            technique="cross_encoder",
            model=self.settings.re_ranker_model_name,
            latency_ms=rerank_latency_ms,
            candidate_count=len(results),
            final_results_count=len(final_results),
            query_count=len(queries),
        )

        return final_results


@lru_cache(maxsize=1)
def get_model():
    settings = get_settings()

    return CrossEncoder(
        settings.re_ranker_model_name,
        max_length=settings.re_ranker_max_length,
    )
