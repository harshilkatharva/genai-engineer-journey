from typing import Literal

from pydantic import BaseModel, Field


class DetailsExtractionModel(BaseModel):
    names: list = Field(description="Names includes in text")
    numbers: list = Field(description="Numbers includes in text")
    locations: list = Field(description="Locations includes in text")


class PromptTestRequest(BaseModel):
    provider: Literal["google", "openai", "anthropic"] = Field(
        description="Name of provider that compare with golden dataset"
    )
