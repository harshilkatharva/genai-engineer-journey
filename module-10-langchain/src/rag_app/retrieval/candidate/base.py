from typing import Protocol
from uuid import UUID

from rag_app.models import RetriveResult


class RetrievalStrategy(Protocol):
    async def retrive(
        self,
        tenant_id: UUID,
        queries: list[str],
        top_k_candidates: int,
    ) -> list[RetriveResult]: ...
