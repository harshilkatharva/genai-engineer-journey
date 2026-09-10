from pydantic import BaseModel, Field


class RAGTextRequest(BaseModel):
    text: str = Field(min_length=1)
