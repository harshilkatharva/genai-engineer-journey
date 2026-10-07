import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response

from ai_app.api.limiter import limiter
from ai_app.features.summarization.service import SummarizationService
from ai_app.models import ChatServiceRequestModel
from ai_app.models.llm_response_model import LLMResponseModel

router = APIRouter()


def get_summarization_services():
    return SummarizationService()


@router.post("/")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    response: Response,
    body: ChatServiceRequestModel,
    background_tasks: BackgroundTasks,
    service: Annotated[SummarizationService, Depends(get_summarization_services)],
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
