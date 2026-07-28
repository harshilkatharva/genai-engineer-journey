import asyncio

from llm_client.models.response_model import CompletionResult
from llm_client.services.providers import AnthropicProvider, OpenAIProvider, GoogleProvider


class LLMClient:
    """
    Connection of all provider
    """

    def __init__(self):
        self.providers = {
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

        return await self.providers[provider].complete(prompt)

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

    async def stream(self, provider: str, prompt: str):
        async for token in self.providers[provider].stream(prompt):
            yield token
