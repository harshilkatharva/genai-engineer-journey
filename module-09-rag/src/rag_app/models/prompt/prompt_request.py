from pydantic import BaseModel


class PromptRequest(BaseModel):
    service: str
    query: str
    chunks: list[str]
