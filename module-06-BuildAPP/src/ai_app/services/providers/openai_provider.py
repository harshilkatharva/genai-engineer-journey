import time
from collections.abc import AsyncIterator

from ai_app.core.config import OPENAI_API_KEY
from ai_app.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai_app.models.llm_response_model import LLMResponseModel
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)


class OpenAIProvider:
    """
    Communicate with OpenAI
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def complete(self, prompt: str) -> LLMResponseModel:
        """
        Get response from OpenAI and send in LLMResponseModel format
        """

        try:
            start = time.perf_counter()

            response = await self.client.responses.create(model="gpt-4o-mini", input=prompt)

            latency = (time.perf_counter() - start) * 1000

            return LLMResponseModel(
                text=response.output_text,
                provider="openai",
                latency_ms=latency,
                token_usage=response.usage.total_tokens if response.usage else 0,
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
        response_stream = await self.client.responses.create(
            model="ggpt-4o-mini", input=prompt, stream=True
        )

        async for event in response_stream:
            if event.type == "response.output_text.delta":
                yield event.delta
