from pydantic import BaseModel
from uuid import UUID


class RetriveResult(BaseModel):
    chunk_text: str
    similarity_score: float


class RetriveResponse(BaseModel):
    tenant_id: UUID
    queries: list[str]
    results: list[RetriveResult]
