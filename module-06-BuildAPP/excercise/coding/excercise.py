import asyncio
import hashlib
import hmac
import logging
import os
import re
import string
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, TypeVar
from uuid import UUID

import psycopg
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

DATABASE_CONNECTION_CONVERSATION_URL = os.environ["DATABASE_CONNECTION_CONVERSATION_URL"]


# 1


def truncate_history(
    self, history: list[dict], recent_turns: int = 10, history_max_token_size: int = 1000
):
    truncated = []
    token_counts = 0
    for message in reversed(history[:recent_turns]):
        if (
            token_counts + message["input_tokens"] + message["output_tokens"]
            > history_max_token_size
        ):
            break
        token_counts += message.input_tokens + message.output_tokens
        truncated.append(message)
    return truncated


# 2


@dataclass
class UsageBreakdown:
    name: str
    message_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost: float


@dataclass
class UsageReport:
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost: float
    total_messages: int
    breakdown: list[UsageBreakdown]


class ClassCalculator:
    def __init__(self):
        self.connection_string = DATABASE_CONNECTION_CONVERSATION_URL

    async def get_user_usage_report(
        self,
        user_id: UUID,
    ) -> UsageReport:
        query = """
            SELECT
                feature,
                COUNT(*) AS message_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost
            FROM history
            WHERE user_id = %s
            GROUP BY feature
            ORDER BY feature
        """

        async with (
            await psycopg.AsyncConnection.connect(self.connection_string) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute(query, (user_id,))
            rows = await cursor.fetchall()

        breakdown = [
            UsageBreakdown(
                name=feature,
                message_count=message_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=float(estimated_cost),
            )
            for (
                feature,
                message_count,
                input_tokens,
                output_tokens,
                estimated_cost,
            ) in rows
        ]

        return UsageReport(
            total_messages=sum(item.message_count for item in breakdown),
            total_input_tokens=sum(item.input_tokens for item in breakdown),
            total_output_tokens=sum(item.output_tokens for item in breakdown),
            total_estimated_cost=sum(item.estimated_cost for item in breakdown),
            breakdown=breakdown,
        )

    async def get_feature_usage_report(
        self,
        feature: str,
    ) -> UsageReport:
        query = """
            SELECT
                user_id,
                COUNT(*) AS message_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost
            FROM history
            WHERE feature = %s
            GROUP BY user_id
            ORDER BY user_id
        """

        async with (
            await psycopg.AsyncConnection.connect(self.connection_string) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute(query, (feature,))
            rows = await cursor.fetchall()

        breakdown = [
            UsageBreakdown(
                name=user_id,
                message_count=message_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=float(estimated_cost),
            )
            for (
                user_id,
                message_count,
                input_tokens,
                output_tokens,
                estimated_cost,
            ) in rows
        ]

        return UsageReport(
            total_messages=sum(item.message_count for item in breakdown),
            total_input_tokens=sum(item.input_tokens for item in breakdown),
            total_output_tokens=sum(item.output_tokens for item in breakdown),
            total_estimated_cost=sum(item.estimated_cost for item in breakdown),
            breakdown=breakdown,
        )

    async def get_day_usage_report(
        self,
        report_date: date,
    ) -> UsageReport:
        query = """
            SELECT
                user_id,
                COUNT(*) AS message_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost
            FROM history
            WHERE created_at::date = %s
            GROUP BY user_id
            ORDER BY user_id
        """

        async with (
            await psycopg.AsyncConnection.connect(self.connection_string) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute(
                query,
                (report_date,),
            )
            rows = await cursor.fetchall()

        breakdown = [
            UsageBreakdown(
                name=user_id,
                message_count=message_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=float(estimated_cost),
            )
            for (
                user_id,
                message_count,
                input_tokens,
                output_tokens,
                estimated_cost,
            ) in rows
        ]

        return UsageReport(
            total_messages=sum(item.message_count for item in breakdown),
            total_input_tokens=sum(item.input_tokens for item in breakdown),
            total_output_tokens=sum(item.output_tokens for item in breakdown),
            total_estimated_cost=sum(item.estimated_cost for item in breakdown),
            breakdown=breakdown,
        )


# 3


T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: float | None = None

        self._lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        """
        Returns whether the primary provider should be called.
        """

        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if (
                    self.opened_at is not None
                    and time.monotonic() - self.opened_at >= self.cooldown_seconds
                ):
                    self.state = CircuitState.HALF_OPEN
                    return True

                return False

            # HALF_OPEN:
            # allow one probe request.
            return True

    async def record_success(self) -> None:
        async with self._lock:
            self.failure_count = 0
            self.opened_at = None
            self.state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()


# 4


class AiConfig(BaseSettings):
    feature_flags: dict[str, bool] = {
        "summarization_v2": True,
        "summarization_cache": True,
    }


class SummarizationService:
    pass


class SummarizationV2Service:
    pass


class AIFeatureDispatcher:
    def __init__(self):
        self.ai_config = AiConfig()

        self.summarization_service = SummarizationService()
        self.summarization_v2_service = SummarizationV2Service()

    async def get_summarization(self):
        if self._is_enabled("summarization_v2"):
            return await self.summarization_v2_service

        return await self.summarization_service

    def _is_enabled(self, flag: str) -> bool:
        return bool(self.ai_config.feature_flags.get(flag, False))


# 5
def normalize_request(text: str) -> str:
    text = text.lower().strip()

    # Remove punctuation.
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse consecutive whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# 6
class ClassificationService:
    pass


class DocumentAIService:
    def __init__(self):
        self.summarization_service = SummarizationService()
        self.classification_service = ClassificationService()

    async def analyze_document(
        self,
        conversation_id: UUID,
        user_id: str,
        request_id: UUID,
        document: str,
    ):
        summary_task = self.summarization_service.get_answer(
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            user_message=document,
        )

        classification_task = self.classification_service.classify(
            document=document,
        )

        summary, classification = await asyncio.gather(
            summary_task,
            classification_task,
        )

        return {
            "summary": summary,
            "classification": classification,
        }


# 7 Not need to test we do not add system message in history so not possible to truncate it. after history trucation we add system prompt


# 8
@dataclass
class UsageRecord:
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost: float
    total_messages: int
    breakdown: list[UsageBreakdown]


def estimate_monthly_cost(records: list[UsageRecord]) -> float:
    now = datetime.now(UTC)
    start = now - timedelta(days=7)

    recent_records = [record for record in records if start <= record.created_at <= now]

    total_cost = sum(float(record.estimated_cost) for record in recent_records)

    daily_average = total_cost / 7

    return daily_average * 30


# 9
@dataclass(frozen=True)
class RateLimitConfig:
    max_requests: int
    window_seconds: int


class AiConfig_Ratelimit:
    def __init__(self):
        self.rate_limits = {
            "free": RateLimitConfig(
                max_requests=10,
                window_seconds=60,
            ),
            "pro": RateLimitConfig(
                max_requests=60,
                window_seconds=60,
            ),
            "enterprise": RateLimitConfig(
                max_requests=300,
                window_seconds=60,
            ),
        }


class RateLimitExceeded(Exception):
    pass


@dataclass
class RateLimitBucket:
    count: int
    window_started_at: float


class AIRateLimiter:
    def __init__(self, ai_config):
        self.ai_config = ai_config

        self._buckets: dict[str, RateLimitBucket] = {}
        self._lock = Lock()

    def check_and_consume(
        self,
        user_id: str,
        subscription_tier: str,
    ) -> None:
        """
        Consume one request for the user.

        Raises RateLimitExceeded when the user's tier quota
        has been exhausted.
        """

        quota = self._get_quota(subscription_tier)

        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(user_id)

            # Start a new window.
            if bucket is None or now - bucket.window_started_at >= quota.window_seconds:
                self._buckets[user_id] = RateLimitBucket(
                    count=1,
                    window_started_at=now,
                )
                return

            if bucket.count >= quota.max_requests:
                retry_after = int(quota.window_seconds - (now - bucket.window_started_at))

                raise RateLimitExceeded(
                    f"Rate limit exceeded for {subscription_tier} tier. "
                    f"Try again in {retry_after} seconds."
                )

            bucket.count += 1

    def _get_quota(self, subscription_tier: str):
        quota = self.ai_config.rate_limits.get(subscription_tier)

        if quota is None:
            raise ValueError(f"Unknown subscription tier: {subscription_tier}")

        return quota


# 10


class LLMClient:
    pass


class CostTracker:
    pass


logger = logging.getLogger("ai.calls")


class AICallLogger:
    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker,
        hash_secret: str,
    ):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.hash_secret = hash_secret.encode()

    def _anonymize_user_id(self, user_id: str) -> str:
        """
        Produce a stable, non-reversible identifier for log correlation.

        HMAC is preferable to a plain SHA256 hash because user IDs may
        be predictable or enumerable.
        """
        return hmac.new(
            self.hash_secret,
            user_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]

    async def complete(
        self,
        *,
        feature: str,
        user_id: str,
        provider: str,
        prompt: str,
        **kwargs: Any,
    ):
        start = time.perf_counter()

        user_hash = self._anonymize_user_id(user_id)

        try:
            response = await self.llm_client.complete(
                provider=provider,
                prompt=prompt,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            input_cost = self.cost_tracker.get_cost(
                input_token=response.input_tokens,
                output_token=0,
                model=response.model,
            )

            output_cost = self.cost_tracker.get_cost(
                input_token=0,
                output_token=response.output_tokens,
                model=response.model,
            )

            total_cost = input_cost + output_cost

            logger.info(
                "ai_call",
                extra={
                    "feature": feature,
                    "user_id": user_hash,
                    "provider": provider,
                    "model": response.model,
                    "latency_ms": round(latency_ms, 2),
                    "cost": float(total_cost),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "status": "success",
                },
            )

            return response

        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000

            logger.exception(
                "ai_call",
                extra={
                    "feature": feature,
                    "user_id": user_hash,
                    "provider": provider,
                    "latency_ms": round(latency_ms, 2),
                    "status": "error",
                },
            )

            raise
