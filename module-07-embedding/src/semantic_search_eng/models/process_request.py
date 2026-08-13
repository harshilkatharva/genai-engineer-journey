from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
