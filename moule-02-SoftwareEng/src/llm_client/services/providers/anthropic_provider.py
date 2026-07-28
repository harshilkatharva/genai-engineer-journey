import time
from typing import AsyncIterator


from llm_client.models.response_model import CompletionResult
from llm_client.config import ANTHROPIC_API_KEY
from anthropic import AsyncAnthropic

class AnthropicProvider:
    """
    Communicate with Anthropic.
    """

    def __init__(self):
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def complete(self, prompt: str) -> CompletionResult:

        start = time.perf_counter()

        response = await self.client.messages.create(
            model="claude-sonnet-4-6",
            messages=[
                {"role" : "user", "content  " : prompt}
            ],
            max_tokens=1024
        )

        latency = (time.perf_counter() - start) * 1000
        total_token = response.usage.input_tokens + response.usage.output_tokens

        return CompletionResult(
            text=response.content[0].text,
            provider="anthropic",
            latency_ms=latency,
            token_usage=total_token,
        )

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream response text deltas from Anthropic
        """
        async with self.client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
        ) as stream:
            async for text in stream.text_stream:
                yield text