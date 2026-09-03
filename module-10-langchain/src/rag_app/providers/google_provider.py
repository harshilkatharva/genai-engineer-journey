import time
from collections.abc import AsyncIterator

from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI

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

        llm = self.llm

        if response_schema:
            llm = self.llm.with_structured_output(response_schema)

        response = await llm.ainvoke(prompt)

        latency = (time.perf_counter() - start) * 1000
        usage = response.usage_metadata or {}
        input_tokens = (
            usage.prompt_token_count if usage and usage.prompt_token_count is not None else 0
        )
        output_tokens = (
            usage.candidates_token_count
            if usage and usage.candidates_token_count is not None
            else 0
        )

        # Structured output response
        if response_schema:
            return LLMResponseModel(
                text=None,
                data=response.model_dump(),
                model=self._get_model(),
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Normal response
        return LLMResponseModel(
            text=response.content,
            data=None,
            model=self._get_model(),
            latency_ms=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # Add exception handling for Google Gemini API errors and raise LLMError with appropriate message

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content

    def _get_model(self) -> str:
        return (
            self.setting.default_llm_model
            if self.setting.default_llm_provider == "google"
            else "gemini-3.5-flash-lite"
        )

    def get_llm(self):
        return self.llm
