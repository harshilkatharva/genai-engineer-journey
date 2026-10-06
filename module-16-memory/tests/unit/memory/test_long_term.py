from __future__ import annotations

from datetime import datetime

import pytest

from memory_system import (
    ConversationMessage,
    LLMMemoryExtractor,
    LongTermMemory,
    MemoryCandidate,
)
from memory_system.core import Settings
from memory_system.memory.extraction import MemoryExtractionError
from memory_system.models import LongTermMemoryRecord
from tests.fakes import FakeLLMProvider, FixedEmbedder, MemoryFakeRepository


# Creates a long-term memory service with deterministic test dependencies.
def create_long_term() -> tuple[MemoryFakeRepository, LongTermMemory]:
    repository = MemoryFakeRepository()
    settings = Settings(memory_embedding_dimension=2)
    return repository, LongTermMemory(repository, FixedEmbedder(), settings=settings)


# Verifies a newer contradictory memory ranks ahead of the superseded record.
@pytest.mark.asyncio
async def test_newer_superseding_memory_ranks_above_old_memory() -> None:
    _, long_term = create_long_term()
    old = await long_term.store(
        "user-1", MemoryCandidate(content="I live in Paris", category="fact")
    )
    newer = await long_term.store(
        "user-1",
        MemoryCandidate(
            content="I moved to Berlin",
            category="fact",
            supersedes_ids=[old.id],
        ),
    )

    results = await long_term.retrieve("user-1", "Where do I live?", limit=2)

    assert results[0].memory.id == newer.id
    assert results[0].superseded is False
    assert results[1].memory.id == old.id
    assert results[1].superseded is True


# Verifies extraction parses safe memories and rejects credential content.
@pytest.mark.asyncio
async def test_extractor_parses_memories_and_rejects_secrets() -> None:
    extractor = LLMMemoryExtractor(
        FakeLLMProvider(
            '[{"content":"Prefers green tea","category":"preference",'
            '"supersedes_ids":[],"expires_at":null}]'
        )
    )
    candidates = await extractor.extract(
        [ConversationMessage(role="user", content="I prefer green tea.")], []
    )

    assert [candidate.content for candidate in candidates] == ["Prefers green tea"]

    unsafe = LLMMemoryExtractor(
        FakeLLMProvider(
            '[{"content":"password is hunter2","category":"fact",'
            '"supersedes_ids":[],"expires_at":null}]'
        )
    )
    with pytest.raises(MemoryExtractionError):
        await unsafe.extract(
            [ConversationMessage(role="user", content="My password is hunter2.")], []
        )


# Verifies direct memory writes reject content containing credentials.
@pytest.mark.asyncio
async def test_direct_storage_rejects_credential_content() -> None:
    _, long_term = create_long_term()

    with pytest.raises(MemoryExtractionError):
        await long_term.store("user-1", MemoryCandidate(content="The password is hunter2"))


# Verifies memory expiration timestamps must specify a timezone.
def test_memory_expiration_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        MemoryCandidate(
            content="temporary preference", expires_at=datetime.fromisoformat("2030-01-01")
        )


# Verifies stored memory creation timestamps must specify a timezone.
def test_long_term_record_requires_timezone_for_creation_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        LongTermMemoryRecord(
            content="temporary preference",
            user_id="user-1",
            created_at=datetime.fromisoformat("2030-01-01"),
        )
