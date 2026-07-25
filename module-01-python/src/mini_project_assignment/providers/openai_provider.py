import time
from mini_project_assignment.models import CompletionResult
from openai import OpenAI
from ..config import OPENAI_API_KEY

class OpenAIProvider:
    """
    Communicate with OpenAI 
    """

    async def complete(self, prompt: str) -> CompletionResult:
        """
        Get response from OpenAI and send in completionresult format
        """

        start = time.perf_counter()

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        latency = (time.perf_counter() - start) * 1000

        return CompletionResult(
            text=response.output_text,
            provider="openai",
            latency_ms=latency,
            token_usage=response.usage.total_tokens
        )
