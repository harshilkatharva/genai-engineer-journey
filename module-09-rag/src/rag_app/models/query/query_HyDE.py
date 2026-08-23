from pydantic import BaseModel, Field


class QueryHyDEModel(BaseModel):
    hypothetical_document: list[str] = Field(
        description="List containing original query followed by 3 semantic variations."
    )
