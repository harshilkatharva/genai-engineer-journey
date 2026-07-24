import time

from ..models import CompletionResult


class AnthropicProvider:
    """
    Communicate with Anthropic.
    """

    async def complete(self, prompt: str) -> CompletionResult:

        start = time.perf_counter()

        # TODO:
        # Call Anthropic API

        latency = (time.perf_counter() - start) * 1000

        return CompletionResult(
            text="Anthropic response",
            provider="anthropic",
            latency_ms=latency,
            token_usage=0,
        )