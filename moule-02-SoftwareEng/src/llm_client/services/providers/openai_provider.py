import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from llm_client.config import OPENAI_API_KEY
from llm_client.models.response_model import CompletionResult


class OpenAIProvider:
    """
    Communicate with OpenAI
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def complete(self, prompt: str) -> CompletionResult:
        """
        Get response from OpenAI and send in completionresult format
        """

        start = time.perf_counter()

        response = await self.client.responses.create(model="gpt-4o-mini", input=prompt)

        latency = (time.perf_counter() - start) * 1000

        return CompletionResult(
            text=response.output_text,
            provider="openai",
            latency_ms=latency,
            token_usage=response.usage.total_tokens,
        )

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response_stream = await self.client.responses.create(
            model="gemini-3.5-flash-lite", contents=prompt, stream=True
        )

        async for event in response_stream:
            if event.type == "response.output_text.delta":
                yield event.delta
