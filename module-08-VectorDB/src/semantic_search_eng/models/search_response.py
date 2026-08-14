from pydantic import BaseModel

from .chunk import Chunk


class SearchResult(BaseModel):
    chunk: Chunk
    similarity_score: float


class SearchResponse(BaseModel):
    tenant_id: str
    query: str
    top_k: int
    results: list[SearchResult]
