from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ai_app.api.limiter import limiter
from ai_app.models import LLMRequestModel
from ai_app.services.llm_service import LLMClient

router = APIRouter()


async def get_llm_client():
    return LLMClient()


@router.post("/")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    response: Response,
    body: LLMRequestModel,
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> Response:
    response = await llm_client.complete(body.provider, body.prompt)

    return response


@router.post("/stream")
@limiter.limit("10/minute")
async def stream(
    request: Request,
    response: Response,
    body: LLMRequestModel,
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> StreamingResponse:
    async def genrator():
        async for chunk in llm_client.stream(body.provider, body.prompt):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(genrator(), media_type="text/event-stream")
