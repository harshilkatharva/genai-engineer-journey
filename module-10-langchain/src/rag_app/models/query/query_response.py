from pydantic import BaseModel


class QueryResponse(BaseModel):
    queries: list[str]
