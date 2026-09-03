from uuid import UUID

from pydantic import BaseModel


class RetriveResult(BaseModel):
    chunk_id: str
    chunk_text: str
    similarity_score: float


class RetriveResponse(BaseModel):
    tenant_id: UUID
    queries: list[str]
    results: list[RetriveResult]
