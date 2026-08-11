import pytest
from ai_app.models.message import Message

from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_start_conversation(conversation_manager):
    conversation_manager.db_operator.create_conversation = AsyncMock(
        return_value="conversation-123"
    )

    result = await conversation_manager.start_conversation("user-123")

    assert result == "conversation-123"

    conversation_manager.db_operator.create_conversation.assert_awaited_once_with("user-123")


@pytest.mark.asyncio
async def test_add_conversation(conversation_manager):
    conversation_manager.db_operator.add_history = AsyncMock()

    await conversation_manager.add_conversation(
        conversation_id="conversation-123",
        user_id="user-123",
        request_id="request-123",
        role="user",
        content="Hello",
        feature="chat",
        llm_model="gpt-5",
        input_tokens=10,
        output_tokens=20,
        estimated_cost=0.01,
        duration_ms=150.5,
    )

    conversation_manager.db_operator.add_history.assert_awaited_once_with(
        "conversation-123",
        "user-123",
        "request-123",
        "user",
        "Hello",
        "chat",
        "gpt-5",
        10,
        20,
        0.01,
        150.5,
    )


@pytest.mark.asyncio
async def test_get_conversations_empty(conversation_manager):
    conversation_manager.db_operator.get_history = AsyncMock(return_value=[])

    result = await conversation_manager.get_conversations("conversation-123")

    assert result == []

    conversation_manager.db_operator.get_history.assert_awaited_once_with("conversation-123")


@pytest.mark.asyncio
async def test_get_conversations_truncates_history(
    conversation_manager,
):
    message_1 = Message(
        role="user",
        content="message 1",
        input_tokens=10,
        output_tokens=20,
    )

    message_2 = Message(
        role="assistant",
        content="message 2",
        input_tokens=20,
        output_tokens=20,
    )

    message_3 = Message(
        role="user",
        content="message 3",
        input_tokens=30,
        output_tokens=20,
    )

    history = [
        message_1,
        message_2,
        message_3,
    ]

    conversation_manager.db_operator.get_history = AsyncMock(return_value=history)

    result = await conversation_manager.get_conversations("conversation-123")

    assert result == [
        message_3,
        message_2,
    ]


def test_truncate_history_newest_message_exceeds_limit(
    conversation_manager,
):
    history = [
        Message(
            role="user",
            content="old message",
            input_tokens=20,
            output_tokens=20,
        ),
        Message(
            role="assistant",
            content="very large message",
            input_tokens=80,
            output_tokens=30,
        ),
    ]

    result = conversation_manager._truncate_history(history)

    assert result == []
