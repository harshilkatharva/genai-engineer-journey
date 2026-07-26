import time

from mini_project_assignment.models import CompletionResult
from mini_project_assignment.config import ANTHROPIC_API_KEY
import anthropic

class AnthropicProvider:
    """
    Communicate with Anthropic.
    """

    async def complete(self, prompt: str) -> CompletionResult:

        start = time.perf_counter()

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            messages=[
                {"role" : "user", "content  " : prompt}
            ],
            max_tokens=1024
        )

        latency = (time.perf_counter() - start) * 1000
        total_token = response.usage.input_tokens + response.usage.output_tokens

        return CompletionResult(
            text=response.content[0].text,
            provider="anthropic",
            latency_ms=latency,
            token_usage=total_token,
        )