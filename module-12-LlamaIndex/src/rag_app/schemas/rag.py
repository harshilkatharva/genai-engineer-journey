from typing import Literal

from pydantic import BaseModel, Field

QueryRoute = Literal["vector", "summary"]


class IngestRequest(BaseModel):
    data_dir: str | None = Field(
        default=None, description="Directory containing the document corpus"
    )


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    route: Literal["vector", "summary", "auto"] = "auto"


class SourceReference(BaseModel):
    source: str
    snippet: str


class IngestResponse(BaseModel):
    source_dir: str
    documents: int
    nodes: int


class QueryResponse(BaseModel):
    answer: str
    route: QueryRoute
    sources: list[SourceReference]


class HealthResponse(BaseModel):
    status: str
    ready: bool
    documents: int
