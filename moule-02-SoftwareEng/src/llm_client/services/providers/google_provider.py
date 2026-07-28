import time
from llm_client.models.response_model import CompletionResult
from llm_client.config import GOOGLE_API_KEY
from google import genai
import asyncio
from typing import Iterator, AsyncIterator

class GoogleProvider:
    """
    Communicate with Google Gemini.
    """

    async def complete(self, prompt: str) -> CompletionResult:

        start = time.perf_counter()

        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        latency = (time.perf_counter() - start) * 1000

        return CompletionResult(
            text=response.text,
            provider="google",
            latency_ms=latency,
            token_usage=response.usage_metadata.total_token_count,
        )   


    async def stream(self, prompt : str) -> AsyncIterator[str]:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response_stream = await client.aio.models.generate_content_stream(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text


async def main():

    google_pro = GoogleProvider()
    async for text_chunk in google_pro.stream("What is AI? Explain me in 2 sentence"):
        print(text_chunk,end="",flush=True)
        print("\n")
    print()

if __name__ == "__main__":
    asyncio.run(main())
    
