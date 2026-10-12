from typing import Literal

from pydantic import BaseModel, Field


class DetailsExtractionModel(BaseModel):
    names: list[str] = Field(description="Names includes in text")
    numbers: list[str] = Field(description="Numbers includes in text")
    locations: list[str] = Field(description="Locations includes in text")


class PromptTestRequest(BaseModel):
    provider: Literal["google", "openai", "anthropic"] = Field(
        description="Name of provider that compare with golden dataset"
    )
