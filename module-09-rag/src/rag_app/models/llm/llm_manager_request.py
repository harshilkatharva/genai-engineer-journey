from pydantic import BaseModel, Field

from rag_app.core.settings import get_settings

settings = get_settings()


class LLMManagerRequest(BaseModel):
    provider: str | None = Field(default=settings.default_llm_provider)
    prompt: str
