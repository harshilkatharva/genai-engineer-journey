import time
from mini_project_assignment.models import CompletionResult

class OpenAIProvider:
    """
    Communicate with OpenAI 
    """

    async def complte(self, prompt: str) -> CompletionResult:
        """
        Get response from OpenAI and send in completionresult format
        """

        start = time.perf_counter()


        latency = (time.perf_counter() - start) * 1000

        return CompletionResult(
            text="OpenAI Reponse",
            provider="openai",
            latency_ms=latency,
            token_usage=0
        )
