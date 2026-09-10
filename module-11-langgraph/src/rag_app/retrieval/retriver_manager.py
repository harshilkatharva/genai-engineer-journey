from __future__ import annotations
import time

from rag_app.core import get_settings
from rag_app.models import RetriveRequest, RetriveResponse, QueryManagerRequest
from rag_app.query.query_manager import QueryManager

from pydantic import Field
from rag_app.observability.logger import logger
from rag_app.observability.events import EventName


from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .candidate.base import RetrievalStrategy
from .candidate.hybrid_search import HybridSearch
from .candidate.keyword_search import KeywordSearch
from .candidate.vector_search import VectorSearch
from .re_ranker.base import Reranker
from .re_ranker.cross_encoder_reranker import CrossEncoderReranker


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
        self.query_manager = QueryManager()
        self.strategies: dict[str, RetrievalStrategy] = {
            "vector_search": VectorSearch(),
            "keyword_search": KeywordSearch(),
            "hybrid_search": HybridSearch(),
        }

        self.reranker: Reranker = CrossEncoderReranker()

    async def retrieve(
        self,
        request: RetriveRequest,
    ) -> RetriveResponse:
        tenant_id = request.tenant_id
        queries = await self._get_quries(request.query)
        top_k_candidate = request.top_k_candidate
        top_k_re_ranker = request.top_k_re_ranker

        if len(queries) == 0:
            raise ValueError("Query cannot be empty.")

        strategy = self.settings.default_retrieval_strategy

        # Strategy retrival
        results = await self.strategies[strategy].retrive(
            tenant_id=tenant_id, queries=queries, top_k_candidates=top_k_candidate
        )

        # re ranker
        if self.settings.re_ranker_availability:
            results = await self.reranker.rerank(
                queries=queries,
                results=results,
                top_k=top_k_re_ranker,
            )

        return RetriveResponse(tenant_id=tenant_id, queries=queries, results=results)

    async def _get_quries(self, query: str) -> list[str]:
        query_start = time.perf_counter()

        queries = await self.query_manager.get_queries(request=QueryManagerRequest(query=query))

        query_latency_ms = (time.perf_counter() - query_start) * 1000

        logger.info(
            "Query processing completed",
            event=EventName.QUERY_COMPLETED,
            component="retrive_manager",
            latency_ms=query_latency_ms,
            no_of_queries=len(queries.queries),
        )

        return queries.queries


class LangchainRetriever(BaseRetriever):
    retriever_manager: RetriverManager = Field(default_factory=RetriverManager)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        metadata = getattr(run_manager, "metadata", None) or {}

        tenant_id = metadata.get("tenant_id")

        if tenant_id is None:
            raise ValueError("tenant_id is required for retrieval.")
        response = await self.retriever_manager.retrieve(
            RetriveRequest(
                tenant_id=tenant_id,
                query=query,
            )
        )

        return [
            Document(
                page_content=result.chunk_text,
                metadata={
                    "chunk_id": str(result.chunk_id),
                    "document_name": result.document_name,
                    "similarity_score": result.similarity_score,
                },
            )
            for result in response.results
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        raise NotImplementedError("This retriever supports async retrieval. Use ainvoke().")
