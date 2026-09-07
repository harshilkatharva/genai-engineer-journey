from pydantic import BaseModel


class QueryPerformanceTracker(BaseModel):
    app_version: str
    query: str
    chunk_ids: list[str] | None
    llm_answer: str
