import time
from collections.abc import AsyncIterator

from langchain_core.messages import UsageMetadata
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from rag_app.core.config import GOOGLE_API_KEY
from rag_app.core.settings import get_settings
from rag_app.models import LLMResponseModel
from rag_app.providers.llm_provider import LLMProvider


class GoogleProvider(LLMProvider):
    """
    Communicate with Google Gemini.
    """

    def __init__(self) -> None:
        self.setting = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=self._get_model(),
            google_api_key=GOOGLE_API_KEY,
            temperature=self.setting.default_llm_temperature,
        )

    async def complete(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponseModel:
        start = time.perf_counter()

        if response_schema:
            llm = self.llm.with_structured_output(response_schema)

            response = await llm.ainvoke(prompt)

            latency = (time.perf_counter() - start) * 1000

            return LLMResponseModel(
                text=None,
                data=response.model_dump() if isinstance(response, BaseModel) else response,
                model=self._get_model(),
                latency_ms=latency,
                input_tokens=0,
                output_tokens=0,
            )

        response = await self.llm.ainvoke(prompt)
        print(response)
        latency = (time.perf_counter() - start) * 1000

        usage: UsageMetadata | dict[str, int] = response.usage_metadata or {}

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        content = response.content
        text: str | None
        if isinstance(content, str):
            text = content
        elif content is None:
            text = None
        else:
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text_value = block.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            text = "".join(parts) if parts else None

        return LLMResponseModel(
            text=text,
            data=None,
            model=self._get_model(),
            latency_ms=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # Add exception handling for Google Gemini API errors and raise LLMError with appropriate message

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self.llm.astream(prompt):
            content = chunk.content
            if content is None:
                continue

            if isinstance(content, str):
                if content:
                    yield content
                continue

            for block in content:
                if isinstance(block, str):
                    if block:
                        yield block
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        yield text

    def _get_model(self) -> str:
        return (
            self.setting.default_llm_model
            if self.setting.default_llm_provider == "google"
            else "gemini-3.5-flash-lite"
        )

    def get_llm(self):
        return self.llm
