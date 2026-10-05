from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from ..models import ChatMessage
from ..models.memory import ConversationMessage, LongTermMemoryRecord, MemoryCandidate
from ..providers.llm_provider import LLMProvider


class MemoryExtractionError(ValueError):
    pass


class MemoryExtractor(Protocol):
    async def extract(
        self,
        messages: list[ConversationMessage],
        existing_memories: list[LongTermMemoryRecord],
    ) -> list[MemoryCandidate]: ...


_SENSITIVE_PATTERNS = (
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_ -]?key|access[_ -]?token)"
        r"\b.{0,16}\b(?:is|=|:)\s*\S+"
    ),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)


def validate_memory_content(text: str) -> None:
    if any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS):
        raise MemoryExtractionError(
            "Refusing to store content containing credential or identifier data"
        )


class LLMMemoryExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self._adapter = TypeAdapter(list[MemoryCandidate])

    async def extract(
        self,
        messages: list[ConversationMessage],
        existing_memories: list[LongTermMemoryRecord],
    ) -> list[MemoryCandidate]:
        existing = [
            {
                "id": str(memory.id),
                "category": memory.category,
                "content": memory.content,
            }
            for memory in existing_memories
        ]
        conversation = [{"role": message.role, "content": message.content} for message in messages]
        response = await self.provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Extract only durable user facts, preferences, goals, and constraints. "
                        "Return a JSON array of objects with content, category, supersedes_ids, "
                        "and expires_at (ISO-8601 or null). For a correction, include the IDs of "
                        "the old memories it replaces in supersedes_ids. Do not extract secrets, "
                        "passwords, financial account details, or sensitive health information. "
                        "Return [] when there is nothing safe and useful to remember."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {"existing_memories": existing, "conversation": conversation},
                        ensure_ascii=True,
                    ),
                ),
            ]
        )
        if not response.text:
            raise MemoryExtractionError("Memory extractor returned no text")
        try:
            payload = json.loads(response.text)
            candidates = self._adapter.validate_python(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise MemoryExtractionError("Memory extractor returned invalid memory data") from exc
        for candidate in candidates:
            validate_memory_content(candidate.content)
        valid_ids = {memory.id for memory in existing_memories}
        if any(
            unknown_id not in valid_ids
            for candidate in candidates
            for unknown_id in candidate.supersedes_ids
        ):
            raise MemoryExtractionError(
                "Extractor referenced a superseded memory that does not exist"
            )
        return candidates
