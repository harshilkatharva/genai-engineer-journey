from pydantic import BaseModel


class LLMResponseModel(BaseModel):
    names: list
    numbers: list
    locations: list
