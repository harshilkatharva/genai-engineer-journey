from pydantic import BaseModel


class RetriveResult(BaseModel):
    chunk_text: str
    similarity_score: float


class RetriveResponse(BaseModel):
    tenant_id: str
    query: str
    top_k: int
    results: list[RetriveResult]
