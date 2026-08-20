from pydantic import BaseModel, Field

from rag_app.config import get_settings
from .candidate_response import RetriveResult

settings = get_settings()


class ReRankerRequest(BaseModel):
    chunks: list[RetriveResult]
    top_k_re_ranker: int = Field(
        default=settings.re_ranker_default_top_k, ge=1, le=settings.re_ranker_max_top_k
    )
