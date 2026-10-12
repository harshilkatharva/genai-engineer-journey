import time
from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from llm_client.config import ANTHROPIC_API_KEY
from llm_client.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm_client.models.response_model import CompletionResult


class AnthropicProvider:
    """
    Communicate with Anthropic.
    """

    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def complete(self, prompt: str) -> CompletionResult:
        try:
            start = time.perf_counter()

            response = await self.client.messages.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            latency = (time.perf_counter() - start) * 1000
            block = response.content[0]
            if block.type == "text":
                text = block.text
            total_token = response.usage.input_tokens + response.usage.output_tokens

            return CompletionResult(
                text=text,
                provider="anthropic",
                latency_ms=latency,
                token_usage=total_token,
            )

        except RateLimitError as e:
            raise LLMRateLimitError(str(e))

        except APIConnectionError as e:
            raise LLMConnectionError(str(e))

        except APITimeoutError as e:
            raise LLMTimeoutError(str(e))

        except AuthenticationError as e:
            raise LLMAuthenticationError(str(e))

        except APIError as e:
            raise LLMError(str(e))

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream response text deltas from Anthropic
        """
        async with self.client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
