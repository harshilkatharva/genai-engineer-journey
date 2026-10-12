from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, ClassVar, Literal

import httpx
import pytest
import tiktoken
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response


# 2
def token_estimate(sentence: str) -> int:
    """
    Estimates the number of tokens in a given sentence using the tiktoken library.

    Args:
        sentence (str): The input sentence for which the token count is to be estimated.

    Returns:
        int: The estimated number of tokens in the input sentence.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(sentence)
    return len(tokens)


class PromptRequest(BaseModel):
    MAX_TOKENS: ClassVar[int] = 1000

    prompt: str = Field(min_length=1, description="Prompt length at least 1")

    @field_validator("prompt")
    @classmethod
    def validate_prompt_token(cls, value: str) -> str:
        token_count = token_estimate(value)

        if token_count > cls.MAX_TOKENS:
            raise ValueError(f"Prompt exceeds {cls.MAX_TOKENS} tokens (estimated: {token_count})")

        return value


app = FastAPI()
logger = logging.getLogger(__name__)


# 3
@app.middleware("http")
async def log_request(request: Request, call_func: RequestResponseEndpoint) -> Response:
    start = time.perf_counter()

    response = await call_func(request)

    latency = time.perf_counter() - start

    logger.info(
        "%s %s %s %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )

    return response


# 4

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "application/pdf",
}


@app.post("/upload")
async def upload_file(
    file: Annotated[UploadFile, File()],
) -> dict[str, str | int | None]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File size exceeds the 5 MB limit",
        )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "message": "File uploaded successfully",
    }


# 5


@app.get("/stream")
async def stream() -> StreamingResponse:
    async def generate() -> AsyncIterator[str]:
        yield "data: Hello\n\n"
        await asyncio.sleep(1)
        yield "data: World\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@pytest.mark.asyncio
async def test_sse_streams_incrementally() -> None:
    start = time.perf_counter()

    async with (
        httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client,
        client.stream("GET", "/stream") as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        first_event = None

        async for line in response.aiter_lines():
            if line.startswith("data:"):
                first_event = line
                break

        time_to_first_byte = time.perf_counter() - start

    assert first_event is not None
    assert time_to_first_byte < 0.5


# 6
async def check_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health_check")
async def health() -> dict[str, str]:
    return await check_health()


# 7
def get_request_id(request: Request) -> str:
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    return request_id


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        logger.exception("Request handling failed")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )


# 8


async def get_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


class LLMTimeoutError(RuntimeError):
    pass


class FakeLLMClient:
    async def complete(self, provider: str, prompt: str) -> dict[str, str]:
        raise LLMTimeoutError("LLM provider timed out")


async def override_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.mark.asyncio
async def test_llm_timeout(api_client: httpx.AsyncClient) -> None:
    app.dependency_overrides[get_llm_client] = override_llm_client

    try:
        response = await api_client.post(
            "/chat",
            json={
                "provider": "openai",
                "prompt": "Hello",
            },
        )

        assert response.status_code == 504

        data = response.json()

        assert data == {
            "detail": "LLM provider timed out",
        }

    finally:
        app.dependency_overrides.clear()


# 9


class TextRequest(BaseModel):
    type: Literal["text"]
    prompt: str


class ChatRequest(BaseModel):
    type: Literal["chat"]
    messages: list[str]


class LLMChatRequest(BaseModel):
    provider: Literal["openai", "google", "anthropic"]
    prompt: str


RequestModel = Annotated[
    TextRequest | ChatRequest,
    Field(discriminator="type"),
]


# 10


def save_chat_interaction(provider: str, prompt: str, response: str) -> None:
    # add loggging method here
    pass


@app.post("/chat")
async def chat(
    request: LLMChatRequest,
    background_tasks: BackgroundTasks,
    llm_client: Annotated[FakeLLMClient, Depends(get_llm_client)],
) -> dict[str, str]:
    result = await llm_client.complete(
        provider=request.provider,
        prompt=request.prompt,
    )

    background_tasks.add_task(
        save_chat_interaction,
        provider=request.provider,
        prompt=request.prompt,
        response=result["text"],
    )

    return result
