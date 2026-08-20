from __future__ import annotations

from rag_app.config import get_settings
from rag_app.models import RetriveRequest, RetriveResponse

from .candidate.vector_search import VectorSearch


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
        self.strategies = {"vector_search": VectorSearch()}

    async def retrieve(
        self,
        request: RetriveRequest,
    ) -> RetriveResponse:
        tenant_id = request.tenant_id
        queries = request.queries
        top_k_candidate = request.top_k_candidate
        # top_k_re_ranker = request.top_k_re_ranker

        if len(queries) == 0:
            raise ValueError("Query cannot be empty.")

        strategy = self.settings.default_retrieval_strategy

        results = await self.strategies[strategy].retrive(
            tenant_id=tenant_id, queries=queries, top_k_candidates=top_k_candidate
        )

        if self.settings.re_ranker_availability:
            # Send to re ranker
            pass

        return RetriveResponse(tenant_id=tenant_id, queries=queries, results=results)
