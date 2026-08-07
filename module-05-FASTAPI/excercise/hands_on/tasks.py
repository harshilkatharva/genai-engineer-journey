import asyncio
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# 1

app = FastAPI()


@app.get("/health")
def health_check():
    # logic for cheking
    return {"message": "Everything working"}


@app.post("/chat")
def chat_llm(request):
    # call llm provider
    return {"answer": "llm answer", "token_usages": "1000"}


# 2
class CheckRequest(BaseModel):
    provider: Literal["openai", "google", "anthropic"] = Field(
        description="Provider name openai, google, anthropic"
    )
    query: str = Field(min_length=1, max_length=500, description="User Query")


class CheckResponse(BaseModel):
    answer: str = Field(min_length=1, description="Answer from provider")
    token_usages: int = Field(ge=1, description="Total Token usage for this process")


# 3

api_key_header = APIKeyHeader(name="API-KEY")
VALID_API_KEY = ["abcusbcduvdsvdcv", "vncbdshvhjbdsvdhs"]


async def verify_api_key(key: str = Security(api_key_header)) -> str:
    if key not in VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


@app.post("/chat_auth")
def chat_llm_auth(request: CheckRequest, api_key: str = Security(verify_api_key)) -> CheckResponse:
    # call llm provider
    return {"answer": "llm answer", "token_usages": "1000"}


# 4
class LLMRateLimitError(Exception):
    # our LLM client share
    pass


class LLMTimeoutError(RuntimeError):
    # our LLM client share
    pass


@app.exception_handler(LLMRateLimitError)
async def rate_limit_handler(request: Request, exc: LLMRateLimitError):
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "message": "Please retry shortly."},
    )


@app.exception_handler(LLMTimeoutError)
async def timeout_handler(request: Request, exc: LLMTimeoutError):
    return JSONResponse(
        status_code=504,
        content={"error": "upstream_timeout", "message": "The AI provider timed out."},
    )


# 5
async def fake_stream():
    sentences = ["Hello", "How", "are", "you"]
    for word in sentences:
        await asyncio.sleep(1)
        yield word + " "


@app.post("/chat/stream")
async def chat_stream(request: CheckRequest) -> CheckResponse:
    async def event_genrator():
        # Provider stream
        # ...

        # demo stream
        async for chunk in fake_stream():
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_genrator(), media_type="text/event-stream")


# 6


class LLMServices:
    def complete(self, provider: str, query: str):
        return {"answer": "Openai Response", "token_usages": "1000"}


def get_llm_services():
    return LLMServices()


@app.post("/chat_overrideable")
async def chat(request: CheckRequest, llm: Annotated[LLMServices, Depends(get_llm_services)]):
    response = await llm.complete(request.provider, request.query)

    return response
