from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, Request, Response

from ai_app.api.limiter import limiter
from ai_app.models import ChatServiceRequestModel

from ai_app.features.summarization.service import SummarizationService

router = APIRouter()


def get_summarization_services():
    return SummarizationService()


@router.post("/")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    response: Response,
    body: ChatServiceRequestModel,
    service: Annotated[SummarizationService, Depends(get_summarization_services)],
) -> Response:
    request_id = uuid.uuid4()

    response = await service.get_answer(
        body.conversation_id,
        user_id=body.user_id,
        request_id=request_id,
        user_message=body.user_message,
    )

    return response
