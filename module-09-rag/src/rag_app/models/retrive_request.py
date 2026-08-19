from pydantic import BaseModel, Field

from rag_app.config import get_settings

settings = get_settings()


class RetriveRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    document_type: str | None = Field(
        default=None,
        min_length=1,
    )
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
