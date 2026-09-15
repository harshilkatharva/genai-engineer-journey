import pytest

from rag_app.routing.classifier import QueryRouter


@pytest.mark.parametrize(
    "query",
    ["summarize the reports", "compare the reports", "what are the overall trends?"],
)
def test_synthesis_queries_use_summary_route(query: str) -> None:
    assert QueryRouter().classify(query) == "summary"


def test_specific_fact_queries_use_vector_route() -> None:
    assert QueryRouter().classify("Who signed the agreement in 2024?") == "vector"
