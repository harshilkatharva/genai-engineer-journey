from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_app.models import LLMManagerRequest, QueryHyDEModel
from rag_app.query.techniques.query_hyde import QueryHyDE


@pytest.fixture
def query_hyde():
    with (
        patch("rag_app.query.techniques.query_hyde.PromptManager") as mock_prompt_manager,
        patch("rag_app.query.techniques.query_hyde.LLMServicemanager") as mock_llm_service_manager,
    ):
        manager = QueryHyDE()

        yield (
            manager,
            mock_prompt_manager.return_value,
            mock_llm_service_manager.return_value,
        )


def test_build_prompt_delegates_to_prompt_manager(
    query_hyde,
):
    manager, prompt_manager, _ = query_hyde

    query = "What is the refund policy?"
    expected_prompt = "Generate a hypothetical document for the query."

    prompt_manager.build_query_HyDE_prompt.return_value = expected_prompt

    result = manager._build_prompt(query=query)

    assert result == expected_prompt

    prompt_manager.build_query_HyDE_prompt.assert_called_once_with(
        query=query,
    )


@pytest.mark.asyncio
async def test_call_llm_returns_hypothetical_document(
    query_hyde,
):
    manager, _, llm_service_manager = query_hyde

    prompt = "Generate a hypothetical document."

    answer = MagicMock()
    answer.data = {
        "hypothetical_document": ["A hypothetical document about refund policies."],
    }

    llm_service_manager.complete = AsyncMock(
        return_value=answer,
    )

    result = await manager._call_llm(
        prompt=prompt,
    )

    assert result == ["A hypothetical document about refund policies."]

    llm_service_manager.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_llm_sends_correct_request(
    query_hyde,
):
    manager, _, llm_service_manager = query_hyde

    prompt = "Generate a hypothetical document."

    answer = MagicMock()
    answer.data = {
        "hypothetical_document": ["Hypothetical document content."],
    }

    llm_service_manager.complete = AsyncMock(
        return_value=answer,
    )

    await manager._call_llm(
        prompt=prompt,
    )

    llm_service_manager.complete.assert_awaited_once_with(
        LLMManagerRequest(
            prompt=prompt,
            response_schema=QueryHyDEModel,
        ),
    )


@pytest.mark.asyncio
async def test_process_query_returns_hypothetical_document(
    query_hyde,
):
    manager, _, _ = query_hyde

    query = "What is the refund policy?"

    expected_prompt = "Generate a hypothetical document for the refund policy."

    expected_result = ["Customers may request a refund according to the company's refund policy."]

    manager._build_prompt = MagicMock(
        return_value=expected_prompt,
    )

    manager._call_llm = AsyncMock(
        return_value=expected_result,
    )

    result = await manager.process_query(
        query=query,
    )

    assert result == expected_result

    manager._build_prompt.assert_called_once_with(
        query=query,
    )

    manager._call_llm.assert_awaited_once_with(
        prompt=expected_prompt,
    )


@pytest.mark.asyncio
async def test_process_query_builds_prompt_and_calls_llm(
    query_hyde,
):
    manager, prompt_manager, llm_service_manager = query_hyde

    query = "What is the cancellation policy?"
    expected_prompt = "Generated HyDE prompt."

    prompt_manager.build_query_HyDE_prompt.return_value = expected_prompt

    answer = MagicMock()
    answer.data = {
        "hypothetical_document": [
            "The cancellation policy allows customers to cancel "
            "their subscription under certain conditions."
        ],
    }

    llm_service_manager.complete = AsyncMock(
        return_value=answer,
    )

    result = await manager.process_query(
        query=query,
    )

    assert result == [
        "The cancellation policy allows customers to cancel "
        "their subscription under certain conditions."
    ]

    prompt_manager.build_query_HyDE_prompt.assert_called_once_with(
        query=query,
    )

    llm_service_manager.complete.assert_awaited_once_with(
        LLMManagerRequest(
            prompt=expected_prompt,
            response_schema=QueryHyDEModel,
        ),
    )


@pytest.mark.asyncio
async def test_call_llm_propagates_llm_exception(
    query_hyde,
):
    manager, _, llm_service_manager = query_hyde

    llm_service_manager.complete = AsyncMock(
        side_effect=RuntimeError("LLM service failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="LLM service failed",
    ):
        await manager._call_llm(
            prompt="Generate a hypothetical document.",
        )

    llm_service_manager.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_query_propagates_llm_exception(
    query_hyde,
):
    manager, prompt_manager, llm_service_manager = query_hyde

    prompt_manager.build_query_HyDE_prompt.return_value = "Generated HyDE prompt."

    llm_service_manager.complete = AsyncMock(
        side_effect=RuntimeError("LLM service failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="LLM service failed",
    ):
        await manager.process_query(
            query="What is the refund policy?",
        )

    llm_service_manager.complete.assert_awaited_once()
