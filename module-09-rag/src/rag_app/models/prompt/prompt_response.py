from pydantic import BaseModel


class PromptResposne(BaseModel):
    prompt: str
