from pydantic import BaseModel

from .candidate_response import RetriveResult


class ReRankerResponse(BaseModel):
    chunks: list[RetriveResult]
