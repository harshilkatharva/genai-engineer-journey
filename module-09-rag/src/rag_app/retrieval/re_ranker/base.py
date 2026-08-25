from __future__ import annotations

from typing import Protocol

from rag_app.models import RetriveResult


class Reranker(Protocol):
    async def rerank(
        self,
        queries: list[str],
        results: list[RetriveResult],
        top_k: int,
    ) -> list[RetriveResult]: ...
