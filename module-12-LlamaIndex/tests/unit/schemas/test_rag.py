from rag_app.schemas.rag import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
)


def test_query_request_validates_route() -> None:
    request = QueryRequest(query="What is the answer?", route="summary")

    assert request.query == "What is the answer?"
    assert request.route == "summary"


def test_response_schemas_serialize_nested_sources() -> None:
    response = QueryResponse(
        answer="answer",
        route="vector",
        sources=[SourceReference(source="report.txt", snippet="evidence")],
    )

    assert response.model_dump() == {
        "answer": "answer",
        "route": "vector",
        "sources": [{"source": "report.txt", "snippet": "evidence"}],
    }
    assert IngestRequest().data_dir is None
    assert IngestResponse(source_dir="/data", documents=1, nodes=2).nodes == 2
    assert HealthResponse(status="ok", ready=True, documents=1).ready is True
