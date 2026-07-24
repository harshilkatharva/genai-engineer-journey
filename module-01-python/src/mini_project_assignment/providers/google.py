import time
from  mini_project_assignment.models import CompletionResult
from mini_project_assignment.config import GOOGLE_API_KEY
from google import genai

class GoogleProvider:
    """
    Communicate with Google Gemini.
    """

    async def complete(self, prompt: str) -> CompletionResult:

        start = time.perf_counter()

        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = await client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        response = response.json()
        latency = (time.perf_counter() - start) * 1000

        return CompletionResult(
            text=response.text,
            provider="google",
            latency_ms=latency,
            token_usage=response.usage_metadata,
        )   