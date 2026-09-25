from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from customer_support_agent.core import get_settings
from customer_support_agent.db import CustomerSupportDB
from customer_support_agent.models import ChatMessage
from customer_support_agent.observability.context import reset_request_id, set_request_id
from customer_support_agent.orchestration import run_tool_chat
from customer_support_agent.providers import AnthropicProvider, GoogleProvider, OpenAIProvider
from customer_support_agent.providers.llm_provider import LLMProvider
from customer_support_agent.tools import ToolContext, build_default_registry


class ChatToolRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    customer_id: str | None = None


class ChatToolResponse(BaseModel):
    text: str
    iterations: int
    exhausted: bool = False
    error: str | None = None


def get_db() -> CustomerSupportDB:
    return CustomerSupportDB()


def get_provider() -> LLMProvider:
    provider_name = get_settings().default_llm_provider.lower()
    providers: dict[str, type[LLMProvider]] = {
        "google": GoogleProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    provider = providers.get(provider_name)
    if provider is None:
        raise ValueError(f"Unsupported provider: {provider_name}")
    return provider()


app = FastAPI(title=get_settings().app_name)


@app.post("/chat_tool", response_model=ChatToolResponse)
async def chat_tool(
    request: ChatToolRequest,
    db: Annotated[CustomerSupportDB, Depends(get_db)],
    provider: Annotated[LLMProvider, Depends(get_provider)],
) -> ChatToolResponse:
    request_token = set_request_id(uuid4())
    try:
        registry = build_default_registry(ToolContext(db=db, customer_id=request.customer_id))
        result = await run_tool_chat(
            request.messages, provider, registry, get_settings().max_tool_iterations
        )
        return ChatToolResponse(
            text=result.text,
            iterations=result.iterations,
            exhausted=result.exhausted,
            error=result.error,
        )
    finally:
        reset_request_id(request_token)
