import asyncio
from collections.abc import AsyncIterator
from rag_app.observability.events import EventName
from rag_app.exceptions.llm_exceptions import LLMError
from rag_app.models import LLMManagerRequest, LLMManagerResponse, LLMResponseModel
from rag_app.providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
from rag_app.observability.logger import logger
from rag_app.providers.llm_provider import LLMProvider


class LLMServicemanager:
    """
    Connection of all provider
    """

    def __init__(self) -> None:
        self.providers: dict[str, LLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider(),
        }

    async def complete(self, request: LLMManagerRequest) -> LLMManagerResponse:
        """
        Provide respose from specific provider
        """

        provider = request.provider
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider {provider}")

        try:
            response = await self.providers[provider].complete(
                prompt=request.prompt, response_schema=request.response_schema
            )
            logger.info(
                "LLM Response Genrated",
                event="llm_response_generated",
                component="llm_services",
                provider=provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

            return LLMManagerResponse(text=response.text, data=response.data)

        except LLMError as e:
            logger.exception(
                "LLM Genration Error",
                event=EventName.LLM_FAILED,
                provider=provider,
                error_type=type(e).__name__,
                error_message=e.user_message,
            )

            return LLMManagerResponse(text=e.user_message)

    async def complete_all(
        self,
        prompt: str,
    ) -> list[LLMResponseModel | BaseException]:
        """
        Query every provider concurrently.
        If one provider fails, the others continue
        executing.
        """

        tasks = [provider.complete(prompt) for provider in self.providers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def stream(self, provider: str, prompt: str) -> AsyncIterator[str]:
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider {provider}")

        async for token in self.providers[provider].stream(prompt):
            yield token
