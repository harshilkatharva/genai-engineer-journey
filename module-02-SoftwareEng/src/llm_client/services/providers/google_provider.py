import time
from collections.abc import AsyncIterator

from google import genai
from google.genai.errors import APIError, ClientError, ServerError

from llm_client.config import GOOGLE_API_KEY
from llm_client.models.response_model import CompletionResult

from llm_client.exceptions import (
    LLMError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
)


class GoogleProvider:
    """
    Communicate with Google Gemini.
    """

    def __init__(self) -> None:
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    async def complete(self, prompt: str) -> CompletionResult:
        try:
            start = time.perf_counter()

            response = await self.client.aio.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
            )

            latency = (time.perf_counter() - start) * 1000
            usage = response.usage_metadata
            token_usage = (
                usage.total_token_count
                if usage is not None and usage.total_token_count is not None
                else 0
            )

            return CompletionResult(
                text=response.text or "",
                provider="google",
                latency_ms=latency,
                token_usage=token_usage,
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
        except Exception as e:
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
