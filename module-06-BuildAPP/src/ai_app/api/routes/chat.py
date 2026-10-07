import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response

from ai_app.api.limiter import limiter
from ai_app.features.chat.service import ChatService
from ai_app.models import ChatServiceRequestModel
from ai_app.models.llm_response_model import LLMResponseModel

router = APIRouter()


def get_chat_services():
    return ChatService()


@router.post("/")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    response: Response,
    body: ChatServiceRequestModel,
    background_tasks: BackgroundTasks,
    service: Annotated[ChatService, Depends(get_chat_services)],
) -> LLMResponseModel:
    request_id = uuid.uuid4()

    result = await service.get_answer(
        body.conversation_id,
        user_id=body.user_id,
        request_id=request_id,
        user_message=body.user_message,
        background_tasks=background_tasks,
    )

    return result


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
