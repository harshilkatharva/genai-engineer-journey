from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, Request, Response

from ai_app.api.limiter import limiter
from ai_app.models import ChatServiceRequestModel

from ai_app.features.chat.service import ChatService

router = APIRouter()


def get_chat_services():
    return ChatService()


@router.post("/")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    response: Response,
    body: ChatServiceRequestModel,
    service: Annotated[ChatService, Depends(get_chat_services)],
) -> Response:
    request_id = uuid.uuid4()

    response = await service.get_answer(
        body.conversation_id,
        user_id=body.user_id,
        request_id=request_id,
        user_message=body.user_message,
    )

    return response


# @router.post("/stream")
# @limiter.limit("10/minute")
# async def stream(
#     request: Request,
#     response: Response,
#     body: LLMRequestModel,
#     llm_client: Annotated[LLMClient, Depends(get_llm_client)],
# ) -> StreamingResponse:
#     async def genrator():
#         async for chunk in llm_client.stream(body.provider, body.prompt):
#             yield f"data: {chunk}\n\n"

#     return StreamingResponse(genrator(), media_type="text/event-stream")
