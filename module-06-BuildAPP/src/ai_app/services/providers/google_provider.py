import time
from collections.abc import AsyncIterator

from google import genai
from google.genai.errors import APIError, ClientError, ServerError

from ai_app.core.AIConfig import AiConfig
from ai_app.core.config import GOOGLE_API_KEY
from ai_app.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
)
from ai_app.models.llm_response_model import LLMResponseModel


class GoogleProvider:
    """
    Communicate with Google Gemini.
    """

    def __init__(self) -> None:
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.ai_config = AiConfig()

    async def complete(self, prompt: str) -> LLMResponseModel:
        try:
            start = time.perf_counter()

            model = (
                self.ai_config.default_model
                if self.ai_config.default_provider == "google"
                else "gemini-3.5-flash-lite"
            )

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=prompt,
            )

            latency = (time.perf_counter() - start) * 1000
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count

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
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text


# async def main():

#     google_pro = GoogleProvider()
#     async for text_chunk in google_pro.stream("What is AI? Explain me in 2 sentence"):
#         print(text_chunk,end="",flush=True)
#         print("\n")
#     print()

# if __name__ == "__main__":
#     asyncio.run(main())
