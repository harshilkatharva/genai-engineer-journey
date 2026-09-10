from pydantic import BaseModel, Field


class QueryExpansionModel(BaseModel):
    queries: list[str] = Field(
        description="List containing original query followed by 3 semantic variations."
    )
