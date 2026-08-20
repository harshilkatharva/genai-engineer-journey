import time
from collections.abc import AsyncIterator

from google import genai
from google.genai.errors import APIError, ClientError, ServerError

from rag_app.core.config import GOOGLE_API_KEY
from rag_app.core.settings import get_settings
from rag_app.exceptions.llm_exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
)
from rag_app.models.llm.llm_response_model import LLMResponseModel


class GoogleProvider:
    """
    Communicate with Google Gemini.
    """

    def __init__(self) -> None:
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.setting = get_settings()

    async def complete(self, prompt: str) -> LLMResponseModel:
        try:
            start = time.perf_counter()

            response = await self.client.aio.models.generate_content(
                model=self._get_model(),
                contents=prompt,
            )

            latency = (time.perf_counter() - start) * 1000
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0

            return LLMResponseModel(
                text=response.text or "",
                model=response.model_version,
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except ClientError as e:
            if e.code == 401 or e.code == 403:
                raise LLMAuthenticationError(str(e))
            elif e.code == 429:
                raise LLMRateLimitError(str(e))
            elif e.code == 408:
                raise TimeoutError(str(e))
            else:
                raise LLMError(str(e))
        except ServerError as e:
            raise LLMConnectionError(str(e))
        except APIError as e:
            raise LLMError(str(e))

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response_stream = await self.client.aio.models.generate_content_stream(
            model=self._get_model(),
            contents=prompt,
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def _get_model(self) -> str:
        return (
            self.setting.default_llm_model
            if self.setting.default_llm_provider == "google"
            else "gemini-3.5-flash-lite"
        )
