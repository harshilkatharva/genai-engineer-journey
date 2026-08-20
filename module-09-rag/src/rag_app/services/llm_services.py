import asyncio
import json
from collections.abc import AsyncIterator

from rag_app.exceptions.llm_exceptions import LLMError
from rag_app.logger.llm_error_logger import llm_error_logger
from rag_app.logger.llm_sucess_logger import llm_sucess_logger
from rag_app.models import LLMManagerRequest, LLMManagerResponse, LLMResponseModel
from rag_app.providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
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
            response = await self.providers[provider].complete(request.prompt)

            llm_sucess_logger.info(
                json.dumps(
                    {"provider": provider, "prompt": request.prompt, "response": str(response)}
                )
            )

            return LLMManagerResponse(text=response.text)

        except LLMError as e:
            llm_error_logger.exception(
                {
                    "provider": provider,
                    "prompt": request.prompt,
                    "error_type": type(e).__name__,
                    "user_error_message": e.user_message,
                    "error": str(e),
                }
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
