import time
from collections.abc import AsyncIterator

from pydantic import BaseModel

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from rag_app.core.config import ANTHROPIC_API_KEY
from rag_app.core.settings import get_settings
from rag_app.exceptions.llm_exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from rag_app.models.llm.llm_response_model import LLMResponseModel
from rag_app.providers.llm_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Communicate with Anthropic.
    """

    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.setting = get_settings()

    async def complete(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponseModel:
        try:
            start = time.perf_counter()

            response = await self.client.messages.create(
                model=self._get_model(),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            latency = (time.perf_counter() - start) * 1000
            block = response.content[0]
            if block.type == "text":
                text = block.text

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            return LLMResponseModel(
                text=text,
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
        """
        Stream response text deltas from Anthropic
        """
        async with self.client.messages.stream(
            model=self._get_model(),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def _get_model(self) -> str:
        return (
            self.setting.default_llm_model
            if self.setting.default_llm_provider == "anthropic"
            else "claude-sonnet-4-6"
        )
