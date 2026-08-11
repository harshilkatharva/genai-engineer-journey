import os
import uuid

import pytest
from fastapi import BackgroundTasks

from ai_app.features.chat.service import ChatService
from ai_app.models.llm_response_model import LLMResponseModel

RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_INTEGRATION_TESTS,
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1",
)
@pytest.mark.asyncio
async def test_chat_end_to_end():
    service = ChatService()

    background_tasks = BackgroundTasks()

    conversation_id = await service.conversation_manager.start_conversation(
        user_id="integration-test-user"
    )

    request_id = uuid.uuid4()

    response = await service.get_answer(
        conversation_id=conversation_id,
        user_id="integration-test-user",
        request_id=request_id,
        user_message="My name is Harshil. What is my name?",
        background_tasks=background_tasks,
    )

    # Verify actual LLM response
    assert isinstance(response, LLMResponseModel)
    assert response.text
    assert response.model
    assert response.input_tokens >= 0
    assert response.output_tokens >= 0

    print("\n========== LLM RESPONSE ==========")
    print(response.text)
    print("===================================")

    # One background task should have been registered
    assert len(background_tasks.tasks) == 1

    # Execute the actual background task.
    # This calls the real ConversationManager -> DBOperator -> database.
    await background_tasks()

    # Read the history back from the real database
    history = await service.conversation_manager.get_conversations(conversation_id=conversation_id)

    # We expect:
    # 1. user message
    # 2. assistant response
    assert len(history) == 2

    print("\n========== DATABASE HISTORY ==========")

    for message in history:
        print(f"{message['role']}: {message['content']}")

    print("======================================")
