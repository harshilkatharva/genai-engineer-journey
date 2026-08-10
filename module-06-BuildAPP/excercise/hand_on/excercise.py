import os
from dataclasses import dataclass
from datetime import UTC, datetime

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings

load_dotenv()
# we have not condig here so load here env variable

# 1


@dataclass
class Message:
    role: str
    content: str
    tokens: int


class ConversationManager:
    def __init__(self, max_history_token: int = 6000):
        self.max_history_token = max_history_token

    def build_context(
        self, system_prompt: str, history: list[Message], new_message: str
    ) -> list[dict]:
        truncated = self._truncate_history(history)

        return (
            [{"role": "system", "content": system_prompt}]
            + [
                {"role": message.role, "content": message.content, "tokens": message.tokens}
                for message in truncated
            ]
            + [{"role": "user", "content": new_message}]
        )

    def _truncate_history(self, history: list[Message]):
        truncated = []
        token_counts = 0

        for message in reversed(history):
            if token_counts + message.tokens > self.max_history_token:
                break

            token_counts += message.tokens
            truncated.append(message)

        return list(reversed(truncated))


# 3


class UsageRecord(BaseModel):
    request_id: str
    user_id: str | None
    features: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_usd: float
    timestamp: datetime


class TrackUsages:
    def __init__(self):
        pass

    def add(self, request_id: str, user_id: str, features: str, response: dict) -> UsageRecord:
        return UsageRecord(
            request_id=request_id,
            user_id=user_id,
            features=features,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_usd=self._calculate_cost(
                response.input_tokens, response.output_tokens, 1.2, 2.5
            ),
            timestamp=datetime.now(UTC),
        )

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        input_price_per_million: float,
        output_price_per_million: float,
    ) -> float:
        input_cost = (input_price_per_million / 1000000) * input_tokens
        output_cost = (output_price_per_million / 1000000) * output_tokens

        return input_cost + output_cost


"""
QUERY = 
SELECT  feature, estimated_usd 
FROM database 
WHERE timestamp >= DATE_TRUNC('week', NOW()) G
ROUP BY feature 
ORDER BY estimated_usd DESC;
"""


# 4


class LLMTimeOutError(Exception):
    pass


class LLMRateLimitError(Exception):
    pass


class LLMClient:
    pass


async def complete_with_fallback(prompt, provider_1, provider_2, llm_client):
    try:
        response = await llm_client.complete(prompt, provider_1)

        return response
    except (LLMRateLimitError, LLMTimeOutError):
        response = await llm_client.complete(prompt, provider_2)

        return response


# 5


class AIConfig(BaseSettings):
    default_model: str = os.environ["DEFAULT_LLM_MODEL_ALL"]
    default_temprature: float = 0.3
    max_conversation_history_tokens: int = 6000
    enable_streaming: bool = True
    fallback_provider: str = os.environ["FALLBACK_LLM_MODEL_ALL"]
    feature_flag: dict[str, bool] = {"enable_rag": False, "enable_agent": False}


# 6


class GenrationError(Exception):
    pass


@dataclass
class RetriveResult:
    documents: list[str]


@dataclass
class GenrationResult:
    answer: str
    partial: bool = False


class FakeRetriver:
    async def retrive(self, query: str) -> RetriveResult:
        return RetriveResult(
            documents=[
                "Python dataclasses reduce boilerplate.",
                "Pydantic models provide validation.",
            ]
        )


class FakeGenrator:
    async def genrate(self, query: str, documents: list[str]) -> GenrationResult:
        return GenrationError("LLM failed")


async def question_answer(query: str):
    retriver = FakeRetriver()
    genrator = FakeGenrator()

    retrival = await retriver.retrive(query)

    try:
        return await genrator.genrate(query, retrival)
    except GenrationError:
        print("Genration failed")

    return GenrationResult(
        answer="I couldn't generate an answer right now. "
        "The relevant information was retrieved successfully, "
        "but the answer generation service failed.",
        partial=True,
    )
