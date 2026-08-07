from fastapi import APIRouter, Depends
from llm_client.services.llm_service import LLMClient
from llm_client.models import LLMRequestModel, LLMResponseModel
from fastapi.responses import StreamingResponse

from typing import Annotated

router = APIRouter()


async def get_llm_client():
    return LLMClient()


@router.post("/")
async def chat(
    request: LLMRequestModel, llm_client: Annotated[LLMClient, Depends(get_llm_client)]
) -> LLMResponseModel:
    response = await llm_client.complete(request.provider, request.prompt)

    return response


@router.post("/stream")
async def stream(
    request: LLMRequestModel, llm_client: Annotated[LLMClient, Depends(get_llm_client)]
) -> StreamingResponse:
    async def genrator():
        async for chunk in llm_client.stream(request.provider, request.prompt):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(genrator(), media_type="text/event-stream")
