from unittest.mock import AsyncMock

import pytest

from hand_on.excercise import (
    ConversationManager,
    LLMTimeOutError,
    Message,
    TrackUsages,
    complete_with_fallback,
)

demo_history = [
    Message("user", "Hello, how are you? I'm Harshil Kareliya.", 500),
    Message("assistant", "Hello Harshil! I'm doing great. How are you doing today?", 300),
    Message(
        "user",
        "I'm doing fine. I'm currently learning Python and working with dataclasses. Can you explain why dataclasses are useful?",
        1500,
    ),
    Message(
        "assistant",
        "Absolutely. Dataclasses are useful when you create classes mainly for storing data. They automatically generate methods like __init__, __repr__, and __eq__, which saves you from writing repetitive code.",
        1500,
    ),
    Message(
        "user",
        "That makes sense. I'm also building a chatbot and I need to maintain conversation history. What is a good way to represent each message?",
        700,
    ),
    Message(
        "assistant",
        "A dataclass is a good choice. You can create a Message class with fields such as role, content, and tokens. Then you can store each user and assistant message as an object inside a history list.",
        800,
    ),
]

# def test_ConversationManager():
#     conversation_manager = ConversationManager()

#     system_prompt = "You are helpful assistant and your task is help me." # real sytem prompt fatched from prompts/ latest version
#     history = demo_history
#     new_message = "Great! I want to use this history to calculate the total number of tokens and make sure the conversation doesn't exceed the model's context limit. How should I approach that?"

#     context = conversation_manager.build_context(system_prompt,history, new_message)

#     total_token_usages = 0
#     for message in context[1:-1]: # first system and last new message
#         total_token_usages += message['tokens']

#     assert total_token_usages <= 6000


class FakePrimaryClient:
    def __init__(self):
        self.calls = []

    async def complete(self, prompt: str, provider: str):
        self.calls.append(provider)

        return {"provider": provider, "text": f"Response came from {provider}"}


class FakeFallbackClient:
    def __init__(self):
        self.calls = []

    async def complete(self, prompt: str, provider: str):
        self.calls.append(provider)

        if provider == "provider_1":
            raise LLMTimeOutError()

        return {"provider": provider, "text": "Fallback response"}


@pytest.mark.asyncio
async def test_primary_response():
    llm_client = FakePrimaryClient()

    response = await complete_with_fallback(
        prompt="Hello", provider_1="provider_1", provider_2="provider_2", llm_client=llm_client
    )

    assert response["provider"] == "provider_1"

    assert llm_client.calls == ["provider_1"]


@pytest.mark.asyncio
async def test_fallback_response():
    llm_client = FakeFallbackClient()

    response = await complete_with_fallback(
        prompt="Hello", provider_1="provider_1", provider_2="provider_2", llm_client=llm_client
    )

    assert response["provider"] == "provider_2"

    assert llm_client.calls == ["provider_1", "provider_2"]


# 7
@pytest.mark.asyncio
async def test_all_features():
    # request
    request = {
        "request_id": "req-123",
        "user_id": "user-456",
        "message": "Explain Python dataclasses",
    }

    # convertation manager

    history = [
        Message(
            role="user",
            content="Hello",
            tokens=10,
        ),
        Message(
            role="assistant",
            content="Hi Harshil!",
            tokens=10,
        ),
    ]

    convesation_manager = ConversationManager(max_history_token=1000)

    context = convesation_manager.build_context(
        system_prompt="You are a helpful Python assistant.",
        history=history,
        new_message=request["message"],
    )

    assert context[0]["role"] == "system"
    assert context[-1]["role"] == "user"
    assert context[-1]["content"] == "Explain Python dataclasses"

    llm_client = AsyncMock()

    llm_client.complete.return_value = {
        "model": "gpt-5",
        "content": "Dataclasses are useful for creating data-holding classes.",
        "input_tokens": 100,
        "output_tokens": 30,
    }

    response = await llm_client.complete(
        context,
        "gpt-5",
    )

    assert response["content"].startswith("Dataclasses")
    assert response["model"] == "gpt-5"

    tracker = TrackUsages()

    usage = tracker.add(
        request_id=request["request_id"],
        user_id=request["user_id"],
        features="chat",
        response=type(
            "LLMResponse",
            (),
            {
                "model": response["model"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
            },
        )(),
    )

    assert usage.request_id == "req-123"
    assert usage.user_id == "user-456"
    assert usage.features == "chat"
    assert usage.model == "gpt-5"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 30
    assert usage.estimated_usd >= 0

    final_response = {
        "request_id": request["request_id"],
        "answer": response["content"],
    }

    assert final_response["request_id"] == "req-123"
    assert final_response["answer"] == ("Dataclasses are useful for creating data-holding classes.")
