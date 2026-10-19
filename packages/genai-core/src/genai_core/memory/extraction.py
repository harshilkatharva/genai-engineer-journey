from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import TypeAdapter, ValidationError

from ..exceptions import MemoryExtractionError
from ..models import ChatMessage, LLMManagerRequest
from ..models.memory_model import (
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCandidate,
    MemoryCategory,
    MemoryExtractionOutput,
)

if TYPE_CHECKING:
    from ..services.llm_services import LLMService

DEFAULT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a long-term memory extraction assistant. "
    "Analyze the conversation and extract durable user facts, preferences, goals, relationship and context "
    "that are useful across future conversations.\n\n"
    "Guidelines:\n"
    "1. Extract ONLY facts, preferences, goals, relationship and context about the user or their system.\n"
    "2. Do NOT extract ephemeral chatter, greetings, or transient assistant responses.\n"
    "3. Do NOT extract sensitive credentials, passwords, API keys, payment details, or personal health secrets.\n"
    "4. If a new fact updates or contradicts an existing memory, reference the ID of the old memory in 'supersedes_ids'.\n"
    "5. Categorize each candidate into: 'fact', 'preference', 'relationship', 'goal', or 'context'.\n"
    "6. If there is nothing safe, useful, and durable to extract, return an empty list of memories."
)

DEFAULT_EXTRACTION_USER_PROMPT_TEMPLATE = (
    "Existing long-term memories for this user:\n"
    "{existing_memories}\n\n"
    "Conversation to analyze:\n"
    "{conversation}\n\n"
    "Extract durable memory candidates matching the requested schema."
)

# Sensitive credential and identifier regex patterns
_SENSITIVE_PATTERNS = (
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_\s-]?key|access[_\s-]?token|auth[_\s-]?token)"
        r"\b.{0,16}(?:\bis\b|=|:)\s*\S+"
    ),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # Credit card
)


def validate_memory_content(text: str) -> None:
    """Validate that text does not contain credential or sensitive identifier patterns.

    Raises:
        MemoryExtractionError: If text matches sensitive patterns.
    """
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            raise MemoryExtractionError(
                "Refusing to extract or store content containing credential or sensitive identifier data."
            )


class MemoryExtractor(Protocol):
    """Protocol for extracting durable memory candidates from conversation history."""

    async def extract(
        self,
        messages: list[ConversationMessage],
        existing_memories: list[LongTermMemoryRecord],
        custom_prompt_template: str | None = None,
        llm_provider: str | None = None,
    ) -> list[MemoryCandidate]:
        """Extract memory candidates from conversation context."""
        ...


class LLMMemoryExtractor:
    """Extracts typed memory candidates using an LLMService with structured validation and safety filters."""

    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service
        self._list_adapter = TypeAdapter(list[MemoryCandidate])

    async def extract(
        self,
        messages: list[ConversationMessage],
        existing_memories: list[LongTermMemoryRecord],
        custom_prompt_template: str | None = None,
        llm_provider: str | None = None,
    ) -> list[MemoryCandidate]:
        """Analyze conversation history and return validated memory candidates."""
        if not messages:
            return []

        # Format existing memories for LLM context
        existing_formatted = [
            {
                "id": str(rec.id),
                "category": str(rec.category),
                "content": rec.content,
            }
            for rec in existing_memories
        ]

        # Format conversation messages
        conversation_formatted = [
            {"role": str(msg.role), "content": msg.content} for msg in messages
        ]

        # Construct system and user prompt
        system_prompt = DEFAULT_EXTRACTION_SYSTEM_PROMPT
        if custom_prompt_template:
            # If custom prompt contains template variables, format it; otherwise use as system prompt
            if (
                "{conversation}" in custom_prompt_template
                or "{existing_memories}" in custom_prompt_template
            ):
                user_content = custom_prompt_template.format(
                    conversation=json.dumps(conversation_formatted, indent=2, ensure_ascii=True),
                    existing_memories=json.dumps(existing_formatted, indent=2, ensure_ascii=True),
                )
            else:
                system_prompt = custom_prompt_template
                user_content = DEFAULT_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                    conversation=json.dumps(conversation_formatted, indent=2, ensure_ascii=True),
                    existing_memories=json.dumps(existing_formatted, indent=2, ensure_ascii=True),
                )
        else:
            user_content = DEFAULT_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                conversation=json.dumps(conversation_formatted, indent=2, ensure_ascii=True),
                existing_memories=json.dumps(existing_formatted, indent=2, ensure_ascii=True),
            )

        request = LLMManagerRequest(
            provider=llm_provider,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_content),
            ],
            response_schema=MemoryExtractionOutput,
        )

        response = await self.llm_service.complete(request)
        if response.error is not None:
            raise MemoryExtractionError(
                f"LLM memory extraction failed ({response.error.code}): {response.error.message}"
            )

        # Parse candidates from response
        candidates = self._parse_candidates(response)

        # Validate candidates
        validated: list[MemoryCandidate] = []
        for cand in candidates:
            cand_content = cand.content.strip() if cand.content else ""
            if not cand_content:
                continue
            validated.append(
                MemoryCandidate(
                    content=cand_content,
                    category=cand.category or MemoryCategory.FACT,
                    confidence=cand.confidence,
                    supersedes_ids=[str(sid).strip() for sid in cand.supersedes_ids if sid],
                    expires_at=cand.expires_at,
                    metadata=cand.metadata or {},
                )
            )

        return validated

    def _parse_candidates(self, response: Any) -> list[MemoryCandidate]:
        """Extract MemoryCandidate list from structured response or raw response text."""
        # 1. Check structured_response
        structured = getattr(response, "structured_response", None)
        if structured is not None:
            if isinstance(structured, MemoryExtractionOutput):
                return structured.memories
            if isinstance(structured, dict) and "memories" in structured:
                try:
                    return self._list_adapter.validate_python(structured["memories"])
                except ValidationError as err:
                    raise MemoryExtractionError(
                        f"Invalid memory candidate structure in structured response: {err}"
                    ) from err

        # 2. Parse response text as JSON
        text = getattr(response, "text", None)
        if not text or not text.strip():
            return []

        cleaned_text = text.strip()
        # Handle markdown ```json code blocks
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError as err:
            raise MemoryExtractionError(
                f"Failed to parse LLM response as JSON: {err}. Raw text: {text[:200]}"
            ) from err

        try:
            if isinstance(payload, list):
                return self._list_adapter.validate_python(payload)
            if isinstance(payload, dict):
                if "memories" in payload and isinstance(payload["memories"], list):
                    return self._list_adapter.validate_python(payload["memories"])
                return [MemoryCandidate.model_validate(payload)]
            raise MemoryExtractionError(
                f"Expected JSON list or dict with 'memories' key, got {type(payload).__name__}"
            )
        except ValidationError as err:
            raise MemoryExtractionError(
                f"Validation error parsing extracted memories: {err}"
            ) from err
