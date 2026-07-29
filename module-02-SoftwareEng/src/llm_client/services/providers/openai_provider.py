import time
from collections.abc import AsyncIterator

from openai import (
    AsyncOpenAI,
    RateLimitError,
    APITimeoutError,
    AuthenticationError,
    APIConnectionError,
)

from llm_client.config import OPENAI_API_KEY
from llm_client.models.response_model import CompletionResult
from llm_client.exceptions import (
    LLMError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)


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

        try:
            start = time.perf_counter()

            response = await self.client.responses.create(model="gpt-4o-mini", input=prompt)

            latency = (time.perf_counter() - start) * 1000

            return CompletionResult(
                text=response.output_text,
                provider="openai",
                latency_ms=latency,
                token_usage=response.usage.total_tokens,
            )
        except RateLimitError as e:
            raise LLMRateLimitError(str(e))

        except APIConnectionError as e:
            raise LLMConnectionError(str(e))

        except APITimeoutError as e:
            raise LLMTimeoutError(str(e))

        except AuthenticationError as e:
            raise LLMAuthenticationError(str(e))

        except Exception as e:
            raise LLMError(str(e))

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response_stream = await self.client.responses.create(
            model="gemini-3.5-flash-lite", contents=prompt, stream=True
        )

        async for event in response_stream:
            if event.type == "response.output_text.delta":
                yield event.delta
