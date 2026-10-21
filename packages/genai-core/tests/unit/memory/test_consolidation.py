from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from genai_core.exceptions import MemoryExtractionError, MemoryValidationError
from genai_core.memory.extraction import (
    validate_memory_content,
)
from genai_core.memory.in_memory_store import InMemoryMemoryStore
from genai_core.models import (
    ConversationMessage,
    LLMError,
    LLMManagerResponse,
    MemoryCandidate,
    MemoryCategory,
    MemoryExtractionOutput,
    MessageRole,
    WorkingContext,
)
from genai_core.services.memory_service import MemoryService


@pytest.fixture
def mock_llm_service() -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def memory_service(mock_llm_service: MagicMock) -> MemoryService:
    store = InMemoryMemoryStore()
    return MemoryService(store=store, llm_service=mock_llm_service)


@pytest.mark.asyncio
async def test_input_validation_context_sources(memory_service: MemoryService) -> None:
    # 1. Neither session_id nor context
    with pytest.raises(
        MemoryValidationError, match="Either session_id or context must be provided"
    ):
        await memory_service.consolidate(user_id="user-1")

    # 2. Both session_id and context
    context = WorkingContext(messages=[ConversationMessage(role=MessageRole.USER, content="Hello")])
    with pytest.raises(MemoryValidationError, match="Cannot provide both session_id and context"):
        await memory_service.consolidate(user_id="user-1", session_id="session-1", context=context)

    # 3. Empty user_id
    with pytest.raises(MemoryValidationError, match="user_id is required"):
        await memory_service.consolidate(user_id="", session_id="session-1")

    # 4. Invalid context type
    with pytest.raises(MemoryValidationError, match="context must be a WorkingContext"):
        await memory_service.consolidate(
            user_id="user-1",
            context="invalid-type",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_missing_llm_service_error() -> None:
    svc = MemoryService(store=InMemoryMemoryStore(), llm_service=None)
    with pytest.raises(MemoryValidationError, match="LLMService is required"):
        await svc.consolidate(
            user_id="user-1",
            context=WorkingContext(
                messages=[ConversationMessage(role=MessageRole.USER, content="Hello")]
            ),
        )


@pytest.mark.asyncio
async def test_empty_messages_returns_empty_result(memory_service: MemoryService) -> None:
    result = await memory_service.consolidate(
        user_id="user-1",
        context=WorkingContext(messages=[]),
    )
    assert result.total_extracted == 0
    assert result.total_processed == 0
    assert len(result.created) == 0


@pytest.mark.asyncio
async def test_consolidate_via_session_id_successful(
    memory_service: MemoryService, mock_llm_service: MagicMock
) -> None:
    user_id = "user-123"
    session_id = "sess-abc"

    # Populate short term turns
    await memory_service.add_message(
        user_id=user_id, session_id=session_id, role="user", content="I prefer dark mode UI."
    )
    await memory_service.add_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content="Understood, I will remember dark mode.",
    )

    # Mock LLM structured response
    output_schema = MemoryExtractionOutput(
        memories=[
            MemoryCandidate(
                category=MemoryCategory.PREFERENCE,
                content="User prefers dark mode UI.",
                confidence=0.95,
            )
        ]
    )
    mock_llm_service.complete.return_value = LLMManagerResponse(
        structured_response=output_schema,
        model="mock-model",
    )

    result = await memory_service.consolidate(user_id=user_id, session_id=session_id)

    assert result.user_id == user_id
    assert result.session_id == session_id
    assert result.total_extracted == 1
    assert len(result.created) == 1
    assert result.created[0].content == "User prefers dark mode UI."
    assert result.created[0].category == MemoryCategory.PREFERENCE

    # Verify message turns were retrieved and passed to LLM
    call_args = mock_llm_service.complete.call_args[0][0]
    assert call_args.response_schema == MemoryExtractionOutput
    assert any("dark mode" in m.content for m in call_args.messages if m.role == "user")


@pytest.mark.asyncio
async def test_consolidate_via_working_context_and_custom_prompt(
    memory_service: MemoryService, mock_llm_service: MagicMock
) -> None:
    user_id = "user-456"
    custom_prompt = "Custom instructions for extraction: Focus only on programming languages."

    context = WorkingContext(
        messages=[
            ConversationMessage(
                role=MessageRole.USER, content="I am building my backend in Python 3.12."
            )
        ]
    )

    mock_llm_service.complete.return_value = LLMManagerResponse(
        text='{"memories": [{"content": "User develops backend in Python 3.12.", "category": "fact", "confidence": 1.0}]}',
        model="mock-model",
    )

    result = await memory_service.consolidate(
        user_id=user_id,
        context=context,
        custom_prompt_template=custom_prompt,
    )

    assert result.user_id == user_id
    assert result.session_id is None
    assert len(result.created) == 1
    assert result.created[0].content == "User develops backend in Python 3.12."

    # Verify custom prompt was applied in system message
    call_args = mock_llm_service.complete.call_args[0][0]
    system_msg = next(m for m in call_args.messages if m.role == "system")
    assert system_msg.content == custom_prompt


@pytest.mark.asyncio
async def test_consolidate_exact_duplicate_skipped(
    memory_service: MemoryService, mock_llm_service: MagicMock
) -> None:
    user_id = "user-dup"

    # Pre-store an existing fact
    existing = await memory_service.remember(
        user_id=user_id,
        content="User lives in San Francisco.",
        category=MemoryCategory.FACT,
    )

    # LLM extracts the exact same fact
    mock_llm_service.complete.return_value = LLMManagerResponse(
        structured_response=MemoryExtractionOutput(
            memories=[
                MemoryCandidate(
                    content="User lives in San Francisco.",
                    category=MemoryCategory.FACT,
                )
            ]
        )
    )

    result = await memory_service.consolidate(
        user_id=user_id,
        context=[
            ConversationMessage(
                role=MessageRole.USER, content="As you know, I live in San Francisco."
            )
        ],
    )

    assert len(result.created) == 0
    assert len(result.skipped) == 1
    assert f"Exact duplicate of existing record {existing.id}" in str(result.skipped[0].reason)


@pytest.mark.asyncio
async def test_consolidate_semantic_duplicate_skipped(mock_llm_service: MagicMock) -> None:
    user_id = "user-sem-dup"
    mock_embedding = MagicMock()
    # Embeddings for identical or near-identical semantics
    vec = [1.0, 0.0, 0.0]
    mock_embedding.embed_query = AsyncMock(return_value=vec)

    store = InMemoryMemoryStore()
    memory_service = MemoryService(
        store=store,
        embedding_service=mock_embedding,
        llm_service=mock_llm_service,
    )

    # Pre-store existing memory with embedding
    await memory_service.remember(
        user_id=user_id,
        content="I use VS Code as my editor.",
        category=MemoryCategory.PREFERENCE,
    )

    # LLM returns candidate
    mock_llm_service.complete.return_value = LLMManagerResponse(
        structured_response=MemoryExtractionOutput(
            memories=[
                MemoryCandidate(
                    content="User uses VS Code for editing.",
                    category=MemoryCategory.PREFERENCE,
                )
            ]
        )
    )

    result = await memory_service.consolidate(
        user_id=user_id,
        context=[
            ConversationMessage(role=MessageRole.USER, content="VS Code is my primary editor.")
        ],
        similarity_threshold=0.90,
    )

    assert len(result.created) == 0
    assert len(result.skipped) == 1
    assert "High semantic similarity" in str(result.skipped[0].reason)


@pytest.mark.asyncio
async def test_consolidate_supersedes_existing_memory(
    memory_service: MemoryService, mock_llm_service: MagicMock
) -> None:
    user_id = "user-update"

    # Pre-store old fact
    old_record = await memory_service.remember(
        user_id=user_id,
        content="User lives in Chicago.",
        category=MemoryCategory.FACT,
    )

    # LLM returns new fact superseding old record ID
    mock_llm_service.complete.return_value = LLMManagerResponse(
        structured_response=MemoryExtractionOutput(
            memories=[
                MemoryCandidate(
                    content="User relocated and now lives in New York.",
                    category=MemoryCategory.FACT,
                    supersedes_ids=[str(old_record.id)],
                )
            ]
        )
    )

    result = await memory_service.consolidate(
        user_id=user_id,
        context=[
            ConversationMessage(role=MessageRole.USER, content="I recently moved to New York.")
        ],
    )

    assert len(result.created) == 0
    assert len(result.updated) == 1
    assert result.updated[0].content == "User relocated and now lives in New York."
    assert str(old_record.id) in result.updated[0].supersedes_ids

    # Verify old record is marked superseded in store
    active_records = await memory_service.long_term.view(user_id=user_id, include_superseded=False)
    assert len(active_records) == 1
    assert active_records[0].content == "User relocated and now lives in New York."

    all_records = await memory_service.long_term.view(user_id=user_id, include_superseded=True)
    superseded = next(r for r in all_records if str(r.id) == str(old_record.id))
    assert superseded.is_superseded is True


@pytest.mark.asyncio
async def test_sensitive_content_validation_and_skipping(
    memory_service: MemoryService, mock_llm_service: MagicMock
) -> None:
    user_id = "user-sec"

    # Direct validation test
    with pytest.raises(MemoryExtractionError, match="Refusing to extract or store"):
        validate_memory_content("My AWS access key is AKIA1234567890ABCDEF")

    with pytest.raises(MemoryExtractionError, match="Refusing to extract or store"):
        validate_memory_content("The user password is secret_password_123")

    # Non-Luhn digit sequences (e.g. timestamps, epoch ms) should not be blocked as credit cards
    validate_memory_content("Session started at timestamp 1697800000000 in UTC.")

    # Valid Luhn card sequence should be blocked
    with pytest.raises(MemoryExtractionError, match="Refusing to extract or store"):
        validate_memory_content("Card number is 4000123456789017")

    # LLM extracts sensitive content
    mock_llm_service.complete.return_value = LLMManagerResponse(
        structured_response=MemoryExtractionOutput(
            memories=[
                MemoryCandidate(
                    content="User API key is sk-abcdef1234567890abcdef",
                    category=MemoryCategory.FACT,
                ),
                MemoryCandidate(
                    content="User prefers working remotely.",
                    category=MemoryCategory.PREFERENCE,
                ),
            ]
        )
    )

    result = await memory_service.consolidate(
        user_id=user_id,
        context=[ConversationMessage(role=MessageRole.USER, content="Here is my API key")],
    )

    # 1 skipped (sensitive), 1 created (safe)
    assert len(result.skipped) == 1
    assert "Sensitive pattern detected" in str(result.skipped[0].reason)
    assert len(result.created) == 1
    assert result.created[0].content == "User prefers working remotely."


@pytest.mark.asyncio
async def test_malformed_llm_response_error(
    memory_service: MemoryService, mock_llm_service: MagicMock
) -> None:
    user_id = "user-err"

    # Case 1: Provider error
    mock_llm_service.complete.return_value = LLMManagerResponse(
        error=LLMError(provider="test", code="rate_limit", message="Rate limit exceeded")
    )

    with pytest.raises(MemoryExtractionError, match="LLM memory extraction failed"):
        await memory_service.consolidate(
            user_id=user_id,
            context=[ConversationMessage(role=MessageRole.USER, content="Hello")],
        )

    # Case 2: Malformed unparseable JSON text
    mock_llm_service.complete.return_value = LLMManagerResponse(
        text="This is not JSON at all: [broken {"
    )

    with pytest.raises(MemoryExtractionError, match="Failed to parse LLM response as JSON"):
        await memory_service.consolidate(
            user_id=user_id,
            context=[ConversationMessage(role=MessageRole.USER, content="Hello")],
        )


@pytest.mark.asyncio
async def test_storage_failure_gracefully_recorded(
    mock_llm_service: MagicMock,
) -> None:
    user_id = "user-fail"
    failing_store = MagicMock()
    failing_store.list_records = AsyncMock(return_value=[])
    failing_store.save_record = AsyncMock(side_effect=RuntimeError("Database connection lost"))

    memory_service = MemoryService(store=failing_store, llm_service=mock_llm_service)

    mock_llm_service.complete.return_value = LLMManagerResponse(
        structured_response=MemoryExtractionOutput(
            memories=[
                MemoryCandidate(
                    content="User prefers dark mode.",
                    category=MemoryCategory.PREFERENCE,
                )
            ]
        )
    )

    result = await memory_service.consolidate(
        user_id=user_id,
        context=[ConversationMessage(role=MessageRole.USER, content="Dark mode please")],
    )

    assert len(result.created) == 0
    assert len(result.failed) == 1
    assert "Storage failure: Database connection lost" in str(result.failed[0].reason)
