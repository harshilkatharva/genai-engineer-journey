from __future__ import annotations

import json

import pytest

from memory_system import ConversationMessage, LLMMemoryExtractor
from memory_system.memory.extraction import MemoryExtractionError, validate_memory_content
from memory_system.models import LLMManagerResponse, LongTermMemoryRecord
from tests.fakes import FakeLLMService


def test_sensitive_content_patterns_are_rejected() -> None:
    sensitive_content = [
        "password is hunter2",
        "passwd: secret-value",
        "secret = abc123",
        "api_key: value123",
        "access-token is token123",
        "sk-abcdefghijklmnop",
        "AKIA1234567890ABCDEF",
        "123-45-6789",
        "4111 1111 1111 1111",
    ]
    for content in sensitive_content:
        with pytest.raises(MemoryExtractionError, match="credential or identifier"):
            validate_memory_content(content)
    validate_memory_content("Prefers short, direct answers and green tea.")


@pytest.mark.asyncio
async def test_extractor_rejects_empty_malformed_and_invalid_payloads() -> None:
    invalid_responses = [
        (None, "no text"),
        ("not json", "invalid memory data"),
        ("{}", "invalid memory data"),
        ('[{"content":"fact","category":"unknown"}]', "invalid memory data"),
    ]
    for text, message in invalid_responses:
        extractor = LLMMemoryExtractor(FakeLLMService(LLMManagerResponse(text=text)))
        with pytest.raises(MemoryExtractionError, match=message):
            await extractor.extract([ConversationMessage(role="user", content="hello")], [])


@pytest.mark.asyncio
async def test_extractor_only_accepts_supersession_ids_from_existing_records() -> None:
    old_memory = LongTermMemoryRecord(user_id="user-1", content="Lives in Paris")
    response = LLMManagerResponse(
        text=json.dumps(
            [
                {
                    "content": "Lives in Berlin",
                    "category": "fact",
                    "supersedes_ids": [str(old_memory.id)],
                    "expires_at": None,
                }
            ]
        )
    )
    extractor = LLMMemoryExtractor(FakeLLMService(response))

    candidate = await extractor.extract([], [old_memory])

    assert candidate[0].supersedes_ids == [old_memory.id]
    unknown = LLMMemoryExtractor(
        FakeLLMService(
            LLMManagerResponse(
                text=json.dumps(
                    [
                        {
                            "content": "Lives in Berlin",
                            "category": "fact",
                            "supersedes_ids": ["00000000-0000-0000-0000-000000000000"],
                            "expires_at": None,
                        }
                    ]
                )
            )
        )
    )
    with pytest.raises(MemoryExtractionError, match="does not exist"):
        await unknown.extract([], [old_memory])
