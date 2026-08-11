from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks

from ai_app.features.sentiment.service import SentimentService
from ai_app.models.llm_response_model import LLMResponseModel


@pytest.fixture
def sentiment_service():
    with patch("ai_app.features.sentiment.service.AiConfig") as mock_ai_config:
        mock_ai_config.return_value.default_provider = "google"

        service = SentimentService()

        yield service


@pytest.mark.asyncio
async def test_get_answer_sends_prompt_to_llm(sentiment_service):
    response = MagicMock(spec=LLMResponseModel)

    response.text = "Positive"
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 10
    response.output_tokens = 5
    response.latency_ms = 100

    history = [
        {
            "role": "user",
            "content": "I absolutely love this product!",
        }
    ]

    sentiment_service.conversation_manager.get_conversations = AsyncMock(return_value=history)

    sentiment_service.llm_client.complete = AsyncMock(return_value=response)

    background_tasks = BackgroundTasks()

    with patch(
        "ai_app.features.sentiment.service.Path.read_text",
        return_value=("Conversation: {{ conversation_history }}\nUser: {{ user_message }}"),
    ):
        result = await sentiment_service.get_answer(
            conversation_id=uuid4(),
            user_id="user-123",
            request_id=uuid4(),
            user_message="I absolutely love this product!",
            background_tasks=background_tasks,
        )

    assert result == response

    sentiment_service.conversation_manager.get_conversations.assert_awaited_once()

    sentiment_service.llm_client.complete.assert_awaited_once()

    call_kwargs = sentiment_service.llm_client.complete.await_args.kwargs

    assert call_kwargs["provider"] == "google"
    assert "I absolutely love this product!" in call_kwargs["prompt"]


@pytest.mark.asyncio
async def test_get_answer_and_background_persistence(
    sentiment_service,
):
    response = MagicMock(spec=LLMResponseModel)

    response.text = "Positive"
    response.model = "gemini-3.5-flash-lite"
    response.input_tokens = 100
    response.output_tokens = 5
    response.latency_ms = 200

    sentiment_service.conversation_manager.get_conversations = AsyncMock(return_value=[])

    sentiment_service.llm_client.complete = AsyncMock(return_value=response)

    sentiment_service.cost_tracker.get_cost = MagicMock(side_effect=[0.01, 0.02])

    sentiment_service.conversation_manager.add_conversation = AsyncMock()

    background_tasks = BackgroundTasks()

    conversation_id = uuid4()
    request_id = uuid4()

    with patch(
        "ai_app.features.sentiment.service.Path.read_text",
        return_value="{{ user_message }}",
    ):
        result = await sentiment_service.get_answer(
            conversation_id=conversation_id,
            user_id="user-123",
            request_id=request_id,
            user_message="I absolutely love this product!",
            background_tasks=background_tasks,
        )

    assert result == response

    assert len(background_tasks.tasks) == 1

    await background_tasks()

    assert sentiment_service.conversation_manager.add_conversation.await_count == 2
