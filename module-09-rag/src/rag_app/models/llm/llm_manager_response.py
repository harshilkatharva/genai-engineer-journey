from pydantic import BaseModel


class LLMManagerResponse(BaseModel):
    text: str
