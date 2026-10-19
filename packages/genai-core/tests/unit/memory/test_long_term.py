from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from genai_core.exceptions import MemoryValidationError
from genai_core.memory.in_memory_store import InMemoryMemoryStore
from genai_core.memory.long_term import LongTermMemory
from genai_core.models.memory_model import MemoryCandidate, MemoryCategory


@pytest.mark.asyncio
async def test_long_term_memory_store_and_search() -> None:
    store = InMemoryMemoryStore()
    mock_embed = AsyncMock()
    # Return 2-dim unit vector
    mock_embed.embed_query.side_effect = lambda text: [1.0, 0.0] if "tea" in text else [0.0, 1.0]

    ltm = LongTermMemory(store=store, embedding_service=mock_embed)
    user_id = "user_preferences"

    # Store preference
    cand = MemoryCandidate(
        content="I prefer Earl Grey tea in the morning",
        category=MemoryCategory.PREFERENCE,
        confidence=0.95,
    )
    rec = await ltm.store(user_id=user_id, candidate=cand)
    assert rec.content == cand.content
    assert rec.category == MemoryCategory.PREFERENCE
    assert rec.embedding == [1.0, 0.0]

    # Search
    results = await ltm.search(user_id=user_id, query="What tea do I like?")
    assert len(results) == 1
    assert results[0].record.content == cand.content
    assert results[0].similarity == pytest.approx(1.0)
    assert results[0].rank_score > 0.9

    # View
    all_recs = await ltm.view(user_id=user_id)
    assert len(all_recs) == 1

    # Delete
    deleted = await ltm.delete(user_id=user_id, record_id=rec.id)
    assert deleted
    assert len(await ltm.view(user_id=user_id)) == 0

    await store.close()


@pytest.mark.asyncio
async def test_long_term_memory_validation_and_expiration() -> None:
    store = InMemoryMemoryStore()
    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.5, 0.5]
    ltm = LongTermMemory(store=store, embedding_service=mock_embed)

    # Empty user
    with pytest.raises(MemoryValidationError):
        await ltm.store(user_id="", candidate="some fact")

    # Empty content
    with pytest.raises(MemoryValidationError):
        await ltm.store(user_id="u1", candidate="")

    # Content too long (> 4000 chars)
    with pytest.raises(MemoryValidationError):
        await ltm.store(user_id="u1", candidate="x" * 4001)

    # Expiration test
    expired_cand = MemoryCandidate(
        content="Temporary discount code",
        expires_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    await ltm.store(user_id="u1", candidate=expired_cand)

    # Search should exclude expired
    results = await ltm.search(user_id="u1", query="discount code")
    assert len(results) == 0

    # Purge
    purged = await ltm.purge_expired(user_id="u1")
    assert purged == 1

    await store.close()
