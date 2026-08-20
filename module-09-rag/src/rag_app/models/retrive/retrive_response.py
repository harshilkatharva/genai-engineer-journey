from pydantic import BaseModel


class RetriveResult(BaseModel):
    chunk_text: str
    similarity_score: float


class RetriveResponse(BaseModel):
    tenant_id: str
    queries: list[str]
    results: list[RetriveResult]
