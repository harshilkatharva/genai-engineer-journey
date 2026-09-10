from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_app.models import LLMManagerRequest, QueryExpansionModel
from rag_app.query.techniques.query_expansion import QueryExpansion


@pytest.fixture
def query_expansion():
    with (
        patch("rag_app.query.techniques.query_expansion.PromptManager") as mock_prompt_manager,
        patch(
            "rag_app.query.techniques.query_expansion.LLMServicemanager"
        ) as mock_llm_service_manager,
    ):
        manager = QueryExpansion()

        yield (
            manager,
            mock_prompt_manager.return_value,
            mock_llm_service_manager.return_value,
        )


def test_build_prompt_delegates_to_prompt_manager(
    query_expansion,
):
    manager, prompt_manager, _ = query_expansion

    query = "What is the refund policy?"
    expected_prompt = "Generate alternative search queries for: What is the refund policy?"

    prompt_manager.build_query_expansion_prompt.return_value = expected_prompt

    result = manager._build_prompt(query=query)

    assert result == expected_prompt

    prompt_manager.build_query_expansion_prompt.assert_called_once_with(
        query=query,
    )


@pytest.mark.asyncio
async def test_call_llm_returns_expanded_queries(
    query_expansion,
):
    manager, _, llm_service_manager = query_expansion

    prompt = "Expand this query into multiple search queries."

    answer = MagicMock()
    answer.data = {
        "queries": [
            "refund policy",
            "how to get a refund",
            "refund eligibility",
        ],
    }

    llm_service_manager.complete = AsyncMock(
        return_value=answer,
    )

    result = await manager._call_llm(prompt=prompt)

    assert result == [
        "refund policy",
        "how to get a refund",
        "refund eligibility",
    ]

    llm_service_manager.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_llm_sends_correct_request(
    query_expansion,
):
    manager, _, llm_service_manager = query_expansion

    prompt = "Expand this query."

    answer = MagicMock()
    answer.data = {
        "queries": [
            "query one",
            "query two",
        ],
    }

    llm_service_manager.complete = AsyncMock(
        return_value=answer,
    )

    await manager._call_llm(prompt=prompt)

    llm_service_manager.complete.assert_awaited_once_with(
        LLMManagerRequest(
            prompt=prompt,
            response_schema=QueryExpansionModel,
        ),
    )


@pytest.mark.asyncio
async def test_process_query_returns_expanded_queries(
    query_expansion,
):
    manager, _, _ = query_expansion

    query = "What is the refund policy?"
    expected_prompt = "Expand this query."
    expected_queries = [
        "refund policy",
        "refund eligibility",
        "how to request a refund",
    ]

    manager._build_prompt = MagicMock(
        return_value=expected_prompt,
    )

    manager._call_llm = AsyncMock(
        return_value=expected_queries,
    )

    result = await manager.process_query(query=query)

    assert result == expected_queries

    manager._build_prompt.assert_called_once_with(
        query=query,
    )

    manager._call_llm.assert_awaited_once_with(
        prompt=expected_prompt,
    )


@pytest.mark.asyncio
async def test_process_query_builds_prompt_from_query(
    query_expansion,
):
    manager, prompt_manager, llm_service_manager = query_expansion

    query = "What is the cancellation policy?"
    expected_prompt = "Generated expansion prompt."

    prompt_manager.build_query_expansion_prompt.return_value = expected_prompt

    answer = MagicMock()
    answer.data = {
        "queries": [
            "cancellation policy",
            "how to cancel",
        ],
    }

    llm_service_manager.complete = AsyncMock(
        return_value=answer,
    )

    result = await manager.process_query(query=query)

    assert result == [
        "cancellation policy",
        "how to cancel",
    ]

    prompt_manager.build_query_expansion_prompt.assert_called_once_with(
        query=query,
    )

    llm_service_manager.complete.assert_awaited_once_with(
        LLMManagerRequest(
            prompt=expected_prompt,
            response_schema=QueryExpansionModel,
        ),
    )


@pytest.mark.asyncio
async def test_call_llm_propagates_llm_exception(
    query_expansion,
):
    manager, _, llm_service_manager = query_expansion

    llm_service_manager.complete = AsyncMock(
        side_effect=RuntimeError("LLM service failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="LLM service failed",
    ):
        await manager._call_llm(
            prompt="Expand this query.",
        )

    llm_service_manager.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_query_propagates_llm_exception(
    query_expansion,
):
    manager, prompt_manager, llm_service_manager = query_expansion

    prompt_manager.build_query_expansion_prompt.return_value = "Expand this query."

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
