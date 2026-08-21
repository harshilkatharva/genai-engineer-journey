import time
from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from rag_app.core.config import OPENAI_API_KEY
from rag_app.core.settings import get_settings
from rag_app.exceptions.llm_exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from rag_app.models.llm.llm_response_model import LLMResponseModel


class OpenAIProvider:
    """
    Communicate with OpenAI
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.setting = get_settings()

    async def complete(self, prompt: str) -> LLMResponseModel:
        """
        Get response from OpenAI and send in LLMResponseModel format
        """

        try:
            start = time.perf_counter()

            response = await self.client.responses.create(model=self._get_model(), input=prompt)

            latency = (time.perf_counter() - start) * 1000

            usage = response.usage

            input_tokens = usage.input_tokens or 0
            output_tokens = usage.output_tokens or 0

            return LLMResponseModel(
                text=response.output_text,
                model=response.model,
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except RateLimitError as e:
            raise LLMRateLimitError(str(e))

        except APITimeoutError as e:
            raise LLMTimeoutError(str(e))

        except APIConnectionError as e:
            raise LLMConnectionError(str(e))

        except AuthenticationError as e:
            raise LLMAuthenticationError(str(e))

        except APIError as e:
            raise LLMError(str(e))

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response_stream = await self.client.responses.create(
            model=self._get_model(), input=prompt, stream=True
        )

        async for event in response_stream:
            if event.type == "response.output_text.delta":
                yield event.delta

    def _get_model(self) -> str:
        return (
            self.setting.default_llm_model
            if self.setting.default_llm_provider == "openai"
            else "gpt-4"
        )
