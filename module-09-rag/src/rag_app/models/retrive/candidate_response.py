from pydantic import BaseModel

from .retrive_response import RetriveResult


class CandidateResponse(BaseModel):
    chunks: list[RetriveResult]
