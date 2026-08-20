from pydantic import BaseModel


class QueryManagerRequest(BaseModel):
    query: str
    technique: str | None = None
