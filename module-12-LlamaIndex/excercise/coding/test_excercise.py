import pytest
from unittest.mock import AsyncMock, MagicMock


async def route_query(query: str, llm) -> str:
    """
    Routes:
        - specific-fact queries -> vector
        - synthesis / overview queries -> summary
    """

    prompt = f"""
    You are a query router for a RAG system.

    Choose exactly one route:

    vector:
        Use for specific factual questions where the answer
        can be found in specific document chunks.

    summary:
        Use for questions asking for an overview, summary,
        synthesis, or understanding of the entire document.

    Query:
    {query}

    Answer only:
    vector or summary
    """

    response = await llm.acomplete(prompt)

    route = response.text.strip().lower()

    if route not in {"vector", "summary"}:
        raise ValueError(f"Invalid route returned: {route}")

    return route


@pytest.fixture
def mock_llm():
    return MagicMock()


def configure_llm(mock_llm, route: str):
    """
    Configure the mocked LLM to return a specific route.
    """

    response = MagicMock()
    response.text = route

    mock_llm.acomplete = AsyncMock(return_value=response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_route",
    [
        ("Who is the CEO?", "vector"),
        ("What is the revenue?", "vector"),
        ("When was the company founded?", "vector"),
        ("Give me an overview of the document.", "summary"),
        ("Summarize the entire document.", "summary"),
        ("What are the main themes?", "summary"),
    ],
)
async def test_query_routing(
    mock_llm,
    query,
    expected_route,
):
    configure_llm(mock_llm, expected_route)

    route = await route_query(
        query=query,
        llm=mock_llm,
    )

    assert route == expected_route
