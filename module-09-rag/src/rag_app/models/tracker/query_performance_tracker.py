from pydantic import BaseModel, Field


class QueryPerformanceTracker(BaseModel):
    app_version: str
    query: str
    no_of_queries: int = Field(ge=1)
    chunk_ids: list[str]
    llm_answer: str
