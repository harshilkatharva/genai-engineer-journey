from __future__ import annotations
from uuid import UUID

from rag_app.core import get_settings
from rag_app.models import RetriveRequest, RetriveResponse


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
        queries = request.queries
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


class LangchainRetriever(BaseRetriever):
    retriever_manager: RetriverManager

    async def _aget_relevant_documents(
        self,
        tenant_id: UUID,
        queries: list[str],
        *,
        run_manager=None,
    ) -> list[Document]:
        response = await self.retriever_manager.retrieve(
            RetriveRequest(
                tenant_id=tenant_id,
                queries=queries,
            )
        )

        return [
            Document(
                page_content=result.chunk_text,
                metadata={
                    "chunk_id": str(result.chunk_id),
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
