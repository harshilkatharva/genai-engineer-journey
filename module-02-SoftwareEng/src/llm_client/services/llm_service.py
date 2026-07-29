import asyncio
import json

from llm_client.services.provider import LLMProvider

from llm_client.models.response_model import CompletionResult
from llm_client.services.providers import AnthropicProvider, OpenAIProvider, GoogleProvider

from collections.abc import AsyncIterator

from llm_client.exceptions import LLMError

from llm_client.utils.llm_sucess_logger import llm_sucess_logger
from llm_client.utils.llm_error_logger import llm_error_logger


class LLMClient:
    """
    Connection of all provider
    """

    def __init__(self) -> None:
        self.providers: dict[str, LLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider(),
        }

    async def complete(self, provider: str, prompt: str) -> CompletionResult:
        """
        Provide respose from specific provider
        """

        provider = provider.lower()
        if provider not in self.providers:
            raise ValueError(f"Unspported provider {provider}")

        try:
            response = await self.providers[provider].complete(prompt)

            llm_sucess_logger.info(
                json.dumps({"provider": provider, "prompt": prompt, "response": str(response)})
            )

            return response
        except LLMError as e:
            llm_error_logger.exception(
                {
                    "provider": provider,
                    "prompt": prompt,
                    "error_type": type(e).__name__,
                    "user_error_message": e.user_message,
                    "error": str(e),
                }
            )

            return CompletionResult(
                text=e.user_message, provider=provider, latency_ms=0.0, token_usage=0
            )

    async def complete_all(
        self,
        prompt: str,
    ) -> list[CompletionResult | BaseException]:
        """
        Query every provider concurrently.
        If one provider fails, the others continue
        executing.
        """

        tasks = [provider.complete(prompt) for provider in self.providers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def stream(self, provider: str, prompt: str) -> AsyncIterator[str]:
        async for token in self.providers[provider].stream(prompt):
            yield token
