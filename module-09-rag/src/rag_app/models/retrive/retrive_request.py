from uuid import UUID

from pydantic import BaseModel, Field

from rag_app.config import get_settings

settings = get_settings()


class RetriveRequest(BaseModel):
    tenant_id: UUID
    queries: list[str] = Field(min_length=1)
    top_k_candidate: int = Field(
        default=settings.canidate_default_top_k, ge=1, le=settings.canidate_max_top_k
    )
    top_k_re_ranker: int = Field(
        default=settings.re_ranker_default_top_k, ge=1, le=settings.re_ranker_max_top_k
    )
