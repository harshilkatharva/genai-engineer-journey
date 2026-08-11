from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks

from ai_app.features.summarization.service import SummarizationService
from ai_app.models.llm_response_model import LLMResponseModel


@pytest.fixture
def summarization_service():
    service = SummarizationService()

    yield service


@pytest.mark.asyncio
async def test_get_answer_sends_prompt_to_llm(summarization_service):
    response = MagicMock(spec=LLMResponseModel)

    response.text = "This is a summary."
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 10
    response.output_tokens = 20
    response.latency_ms = 100

    history = [
        {
            "role": "user",
            "content": "Python is a programming language.",
        }
    ]

    summarization_service.conversation_manager.get_conversations = AsyncMock(return_value=history)

    summarization_service.llm_client.complete = AsyncMock(return_value=response)

    background_tasks = BackgroundTasks()

    with patch(
        "ai_app.features.summarization.service.Path.read_text",
        return_value=("Conversation: {{ conversation_history }}\nUser: {{ user_message }}"),
    ):
        result = await summarization_service.get_answer(
            conversation_id=uuid4(),
            user_id="user-123",
            request_id=uuid4(),
            user_message="Summarize this text.",
            background_tasks=background_tasks,
        )

    assert result == response

    summarization_service.conversation_manager.get_conversations.assert_awaited_once()

    summarization_service.llm_client.complete.assert_awaited_once()

    call_kwargs = summarization_service.llm_client.complete.await_args.kwargs

    assert call_kwargs["provider"] == "google"
    assert "Python is a programming language." in call_kwargs["prompt"]
    assert "Summarize this text." in call_kwargs["prompt"]


@pytest.mark.asyncio
async def test_get_answer_and_background_persistence(
    summarization_service,
):
    response = MagicMock(spec=LLMResponseModel)

    response.text = "This is a summary."
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 100
    response.output_tokens = 20
    response.latency_ms = 200

    summarization_service.conversation_manager.get_conversations = AsyncMock(return_value=[])

    summarization_service.llm_client.complete = AsyncMock(return_value=response)

    summarization_service.cost_tracker.get_cost = MagicMock(side_effect=[0.01, 0.02])

    summarization_service.conversation_manager.add_conversation = AsyncMock()

    background_tasks = BackgroundTasks()

    conversation_id = uuid4()
    request_id = uuid4()

    with patch(
        "ai_app.features.summarization.service.Path.read_text",
        return_value="{{ user_message }}",
    ):
        result = await summarization_service.get_answer(
            conversation_id=conversation_id,
            user_id="user-123",
            request_id=request_id,
            user_message="Summarize this text.",
            background_tasks=background_tasks,
        )

    assert result == response

    assert len(background_tasks.tasks) == 1

    await background_tasks()

    assert summarization_service.conversation_manager.add_conversation.await_count == 2
