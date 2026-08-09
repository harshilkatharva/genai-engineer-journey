from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from llm_client.api.limiter import limiter
from llm_client.models import PromptTestRequest
from llm_client.services.llm_service import LLMClient
from llm_client.utils.prompt_test import PromptTest

router = APIRouter()


def get_prompt_test() -> PromptTest:
    client = LLMClient()
    return PromptTest(client)


@router.post("/test")
@limiter.limit("2/minute")
async def prompt_test_golden_set(
    request: Request,
    response: Response,
    body: PromptTestRequest,
    prompt_test: Annotated[PromptTest, Depends(get_prompt_test)],
) -> dict:
    await prompt_test.process(body.provider)
    result = prompt_test.check_result()
    return result
