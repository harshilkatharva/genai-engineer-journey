from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from ai_app.features.chat.service import ChatService
from ai_app.models.llm_response_model import LLMResponseModel


@pytest.fixture
def chat_service():
    with (
        patch("ai_app.features.chat.service.AiConfig") as mock_ai_config,
    ):
        mock_ai_config.return_value.default_provider = "google"

        service = ChatService()

        yield service


@pytest.mark.asyncio
async def test_get_answer(chat_service):
    response = MagicMock(spec=LLMResponseModel)

    response.content = "Hello Harshil!"
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 10
    response.output_tokens = 20
    response.latency_ms = 150.5

    chat_service.conversation_manager.get_conversations = AsyncMock(return_value=[])

    chat_service.llm_client.complete = AsyncMock(return_value=response)

    background_tasks = BackgroundTasks()

    result = await chat_service.get_answer(
        conversation_id="conversation-123",
        user_id="user-123",
        request_id="request-123",
        user_message="Hello",
        background_tasks=background_tasks,
    )

    assert result == response

    chat_service.conversation_manager.get_conversations.assert_awaited_once_with(
        conversation_id="conversation-123"
    )

    chat_service.llm_client.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_answer_sends_prompt_to_llm(chat_service):
    response = MagicMock(spec=LLMResponseModel)

    response.content = "Hello!"
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 10
    response.output_tokens = 20
    response.latency_ms = 100

    history = [
        {
            "role": "user",
            "content": "My name is Harshil",
        }
    ]

    chat_service.conversation_manager.get_conversations = AsyncMock(return_value=history)

    chat_service.llm_client.complete = AsyncMock(return_value=response)

    background_tasks = BackgroundTasks()

    with patch(
        "ai_app.features.chat.service.Path.read_text",
        return_value=("Conversation: {{ conversation_history }}\nUser: {{ user_message }}"),
    ):
        await chat_service.get_answer(
            conversation_id="conversation-123",
            user_id="user-123",
            request_id="request-123",
            user_message="What is my name?",
            background_tasks=background_tasks,
        )

    chat_service.llm_client.complete.assert_awaited_once()

    call_kwargs = chat_service.llm_client.complete.await_args.kwargs

    assert call_kwargs["provider"] == "google"

    assert "My name is Harshil" in call_kwargs["prompt"]
    assert "What is my name?" in call_kwargs["prompt"]


@pytest.mark.asyncio
async def test_get_answer_adds_background_task(chat_service):
    response = MagicMock(spec=LLMResponseModel)

    response.content = "Hello!"
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 10
    response.output_tokens = 20
    response.latency_ms = 100

    chat_service.conversation_manager.get_conversations = AsyncMock(return_value=[])

    chat_service.llm_client.complete = AsyncMock(return_value=response)

    background_tasks = BackgroundTasks()

    with patch(
        "ai_app.features.chat.service.Path.read_text",
        return_value="{{ user_message }}",
    ):
        await chat_service.get_answer(
            conversation_id="conversation-123",
            user_id="user-123",
            request_id="request-123",
            user_message="Hello",
            background_tasks=background_tasks,
        )

    assert len(background_tasks.tasks) == 1

    task = background_tasks.tasks[0]

    assert task.func == chat_service._add_conversations

    assert task.kwargs["conversation_id"] == "conversation-123"
    assert task.kwargs["user_id"] == "user-123"
    assert task.kwargs["request_id"] == "request-123"
    assert task.kwargs["user_message"] == "Hello"
    assert task.kwargs["response"] == response


@pytest.mark.asyncio
async def test_get_answer_and_background_persistence(
    chat_service,
):
    response = MagicMock(spec=LLMResponseModel)

    response.text = "Hello Harshil!"
    response.model = "gpt-5"
    response.input_tokens = 100
    response.output_tokens = 50
    response.latency_ms = 200

    chat_service.conversation_manager.get_conversations = AsyncMock(return_value=[])

    chat_service.llm_client.complete = AsyncMock(return_value=response)

    chat_service.cost_tracker.get_cost = MagicMock(side_effect=[0.01, 0.02])

    chat_service.conversation_manager.add_conversation = AsyncMock()

    background_tasks = BackgroundTasks()

    with patch(
        "ai_app.features.chat.service.Path.read_text",
        return_value="{{ user_message }}",
    ):
        result = await chat_service.get_answer(
            conversation_id="conversation-123",
            user_id="user-123",
            request_id="request-123",
            user_message="Hello",
            background_tasks=background_tasks,
        )

    # Response is returned immediately
    assert result == response

    # Background task was registered
    assert len(background_tasks.tasks) == 1

    # Execute background tasks
    await background_tasks()

    # Both messages should now be persisted
    assert chat_service.conversation_manager.add_conversation.await_count == 2
