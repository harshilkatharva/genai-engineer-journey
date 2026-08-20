from pydantic import BaseModel, Field

from rag_app.models.retrive.retrive_response import RetriveResult


class PromptRequest(BaseModel):
    service: str = Field(default="rag_chat")
    query: str
    chunks: list[RetriveResult]
