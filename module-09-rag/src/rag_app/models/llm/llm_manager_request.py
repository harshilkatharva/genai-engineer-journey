from pydantic import BaseModel


class LLMManagerRequest(BaseModel):
    provider: str | None = None
    prompt: str
