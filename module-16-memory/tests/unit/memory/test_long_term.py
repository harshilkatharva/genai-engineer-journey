from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest

from memory_system import (
    ConversationMessage,
    LLMMemoryExtractor,
    LongTermMemory,
    MemoryCandidate,
)
from memory_system.core import Settings
from memory_system.memory.extraction import MemoryExtractionError
from memory_system.models import LLMError, LLMManagerResponse
from tests.fakes import FakeLLMService, FixedEmbedder, MemoryFakeRepository, make_settings


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
        FakeLLMService(
            LLMManagerResponse(
                text='[{"content":"Prefers green tea","category":"preference",'
                '"supersedes_ids":[],"expires_at":null}]',
                model="fake",
            )
        )
    )
    candidates = await extractor.extract(
        [ConversationMessage(role="user", content="I prefer green tea.")], []
    )

    assert [candidate.content for candidate in candidates] == ["Prefers green tea"]

    unsafe = LLMMemoryExtractor(
        FakeLLMService(
            LLMManagerResponse(
                text='[{"content":"password is hunter2","category":"fact",'
                '"supersedes_ids":[],"expires_at":null}]',
                model="fake",
            )
        )
    )
    with pytest.raises(MemoryExtractionError):
        await unsafe.extract(
            [ConversationMessage(role="user", content="My password is hunter2.")], []
        )

    failed = LLMMemoryExtractor(
        FakeLLMService(
            LLMManagerResponse(
                error=LLMError(
                    provider="fake",
                    code="provider_unavailable",
                    message="The provider is unavailable.",
                )
            )
        )
    )
    with pytest.raises(MemoryExtractionError, match="provider_unavailable"):
        await failed.extract([ConversationMessage(role="user", content="I prefer tea.")], [])


@pytest.mark.asyncio
async def test_store_validates_user_supersession_ownership_and_embedding_shape() -> None:
    repository, long_term = create_long_term()

    with pytest.raises(ValueError, match="user_id"):
        await long_term.store("", MemoryCandidate(content="A fact"))
    with pytest.raises(ValueError, match="this user's existing memories"):
        await long_term.store(
            "user-1", MemoryCandidate(content="A correction", supersedes_ids=[uuid4()])
        )
    with pytest.raises(MemoryExtractionError, match="credential or identifier"):
        await long_term.store("user-1", MemoryCandidate(content="The password is hunter2"))

    class WrongDimensionEmbedder:
        async def embed(self, text: str) -> list[float]:
            return [1.0]

    invalid = LongTermMemory(
        repository,
        WrongDimensionEmbedder(),
        settings=make_settings(memory_embedding_dimension=2),
    )
    with pytest.raises(ValueError, match="1 dimensions"):
        await invalid.store("user-1", MemoryCandidate(content="A fact"))
    assert repository.memories == {}


@pytest.mark.asyncio
async def test_store_rejects_non_numeric_or_non_finite_embeddings() -> None:
    invalid_vectors: list[list[object]] = [
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [True, 0.0],
        ["1", 0.0],
    ]
    for vector in invalid_vectors:

        class InvalidEmbedder:
            def __init__(self, values: list[object]) -> None:
                self.values = values

            async def embed(self, text: str) -> list[float]:
                return cast(list[float], self.values)

        long_term = LongTermMemory(
            MemoryFakeRepository(),
            InvalidEmbedder(vector),
            settings=make_settings(memory_embedding_dimension=2),
        )
        with pytest.raises(ValueError, match="non-finite or non-numeric"):
            await long_term.store("user-1", MemoryCandidate(content="A fact"))


@pytest.mark.asyncio
async def test_remember_conversation_requires_extractor_and_valid_supersession_ids() -> None:
    repository, long_term = create_long_term()
    messages = [ConversationMessage(role="user", content="I moved.")]

    with pytest.raises(RuntimeError, match="extractor is required"):
        await long_term.remember_conversation("user-1", messages)

    class UnknownSupersessionExtractor:
        async def extract(self, messages, existing_memories):
            return [MemoryCandidate(content="New fact", supersedes_ids=[uuid4()])]

    long_term.extractor = UnknownSupersessionExtractor()
    with pytest.raises(ValueError, match="Extracted supersedes_ids"):
        await long_term.remember_conversation("user-1", messages)
    assert repository.memories == {}

    class CandidateExtractor:
        async def extract(self, messages, existing_memories):
            return [MemoryCandidate(content="One"), MemoryCandidate(content="Two")]

    long_term.extractor = CandidateExtractor()
    records = await long_term.remember_conversation("user-1", messages)

    assert [record.content for record in records] == ["One", "Two"]
    assert await long_term.view("user-1", 1) == [records[1]]
    for limit in (0, 501):
        with pytest.raises(ValueError, match="between 1 and 500"):
            await long_term.view("user-1", limit)
    with pytest.raises(ValueError, match="user_id"):
        await long_term.view("", 1)
    assert await long_term.delete("user-1", records[0].id) is True
    assert await long_term.delete("user-1", uuid4()) is False
