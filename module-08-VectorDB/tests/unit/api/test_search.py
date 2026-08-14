from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_module(monkeypatch):
    """
    Import the API route module with all expensive managers mocked.

    The route module creates manager instances at import time, so the
    constructors must be patched before importing it.
    """

    import semantic_search_eng.embedding.embedding_manager as embedding_module
    import semantic_search_eng.retrival.retriver_manager as retrival_module

    # mock_embedding_manager_class = MagicMock()
    # mock_retriver_manager_class = MagicMock()

    monkeypatch.setattr(
        embedding_module.SentenceTransformer,
        "__new__",
        staticmethod(lambda cls, *args, **kwargs: MagicMock()),
    )

    monkeypatch.setattr(
        embedding_module.EmbeddingManager,
        "__init__",
        lambda self: None,
    )

    monkeypatch.setattr(
        retrival_module.RetriverManager,
        "__init__",
        lambda self: None,
    )

    modules_to_remove = [
        "semantic_search_eng.api.routes.search",
        "semantic_search_eng.api.app",
    ]

    for module_name in modules_to_remove:
        sys.modules.pop(module_name, None)

    search_module = importlib.import_module("semantic_search_eng.api.routes.search")

    app_module = importlib.import_module("semantic_search_eng.api.app")

    return app_module, search_module


@pytest.fixture
def client(api_module):
    app_module, _ = api_module
    return TestClient(app_module.app)


@pytest.fixture
def mocked_dependencies(api_module):
    _, search_module = api_module

    search_module.data_processor = MagicMock()
    search_module.data_manager = MagicMock()
    search_module.chunking_manager = MagicMock()
    search_module.embedding_manager = MagicMock()
    search_module.retriver_manager = MagicMock()

    return {
        "data_processor": search_module.data_processor,
        "data_manager": search_module.data_manager,
        "chunking_manager": search_module.chunking_manager,
        "embedding_manager": search_module.embedding_manager,
        "retriver_manager": search_module.retriver_manager,
    }


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "Semantic Search Engine"
    assert body["version"] == "0.1.0"
    assert body["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_process_documents_success(
    client,
    mocked_dependencies,
):
    dependencies = mocked_dependencies

    dependencies["data_processor"].process_documents.return_value = [
        "document_0000",
        "document_0001",
    ]

    dependencies["data_manager"].get_document.side_effect = [
        "First document.",
        "Second document.",
    ]

    dependencies["chunking_manager"].chunk_documents.return_value = [
        {
            "document_id": "document_0000",
        }
    ]

    from semantic_search_eng.models.chunk import Chunk

    chunks = [
        Chunk(
            chunk_id="document_0000_chunk_0000",
            document_id="document_0000",
            tenant_id="conversation_001",
            chunk_index=0,
            text="First document.",
            token_count=2,
        ),
        Chunk(
            chunk_id="document_0001_chunk_0000",
            document_id="document_0001",
            tenant_id="conversation_001",
            chunk_index=0,
            text="Second document.",
            token_count=2,
        ),
    ]

    dependencies["chunking_manager"].chunk_documents.return_value = chunks

    dependencies["embedding_manager"].embed_chunks.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    payload = {
        "tenant_id": "conversation_001",
        "documents": [
            "First document.",
            "Second document.",
        ],
    }

    response = client.post(
        "/documents/process",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["tenant_id"] == "conversation_001"
    assert body["document_count"] == 2
    assert body["chunk_count"] == 2
    assert body["embedding_count"] == 2

    assert body["document_ids"] == [
        "document_0000",
        "document_0001",
    ]

    dependencies["data_processor"].process_documents.assert_called_once_with(
        tenant_id="conversation_001",
        documents=[
            "First document.",
            "Second document.",
        ],
    )

    dependencies["chunking_manager"].chunk_documents.assert_called_once()

    dependencies["embedding_manager"].embed_chunks.assert_called_once_with(
        tenant_id="conversation_001",
        chunks=chunks,
        save=True,
    )


def test_process_documents_validation_error(
    client,
    mocked_dependencies,
):
    response = client.post(
        "/documents/process",
        json={
            "tenant_id": "",
            "documents": [],
        },
    )

    assert response.status_code == 422

    mocked_dependencies["data_processor"].process_documents.assert_not_called()


def test_process_documents_internal_error(
    client,
    mocked_dependencies,
):
    mocked_dependencies["data_processor"].process_documents.side_effect = RuntimeError(
        "processing failed"
    )

    response = client.post(
        "/documents/process",
        json={
            "tenant_id": "conversation_001",
            "documents": [
                "document content",
            ],
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == ("processing failed")


def test_search_success(
    client,
    mocked_dependencies,
):
    dependencies = mocked_dependencies

    from semantic_search_eng.models.chunk import Chunk
    from semantic_search_eng.models.search_response import (
        SearchResult,
    )

    chunks = [
        Chunk(
            chunk_id="document_0000_chunk_0000",
            document_id="document_0000",
            tenant_id="conversation_001",
            chunk_index=0,
            text="Authentication information.",
            token_count=2,
        ),
        Chunk(
            chunk_id="document_0000_chunk_0001",
            document_id="document_0000",
            tenant_id="conversation_001",
            chunk_index=1,
            text="Database information.",
            token_count=2,
        ),
    ]

    dependencies["data_manager"].base_path = MagicMock()

    dependencies["retriver_manager"].retrieve.return_value = [
        SearchResult(
            chunk=chunks[0],
            similarity_score=0.95,
        ),
        SearchResult(
            chunk=chunks[1],
            similarity_score=0.41,
        ),
    ]

    # Patch the route helper functions directly so this test focuses
    # on the API contract rather than filesystem implementation.
    import semantic_search_eng.api.routes.search as search_module

    search_module._load_all_chunks = MagicMock(return_value=chunks)

    search_module._load_all_embeddings = MagicMock(
        return_value=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    response = client.post(
        "/search",
        json={
            "tenant_id": "conversation_001",
            "query": "How does authentication work?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["tenant_id"] == ("conversation_001")

    assert body["query"] == ("How does authentication work?")

    assert body["top_k"] == 2

    assert len(body["results"]) == 2

    assert body["results"][0]["chunk"]["chunk_id"] == "document_0000_chunk_0000"

    assert body["results"][0]["similarity_score"] == 0.95

    dependencies["retriver_manager"].retrieve.assert_called_once_with(
        tenant_id="conversation_001",
        query="How does authentication work?",
        chunks=chunks,
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        top_k=2,
    )


def test_search_uses_default_top_k_when_omitted(
    client,
    mocked_dependencies,
):
    dependencies = mocked_dependencies

    from semantic_search_eng.api.routes import search as search_module

    search_module._load_all_chunks = MagicMock(return_value=[])

    search_module._load_all_embeddings = MagicMock(return_value=[])

    response = client.post(
        "/search",
        json={
            "tenant_id": "conversation_001",
            "query": "test query",
        },
    )

    assert response.status_code == 200

    dependencies["retriver_manager"].retrieve.assert_called_once_with(
        tenant_id="conversation_001",
        query="test query",
        chunks=[],
        embeddings=[],
        top_k=5,
    )

    assert response.json()["top_k"] == 5


def test_search_rejects_invalid_request(
    client,
    mocked_dependencies,
):
    response = client.post(
        "/search",
        json={
            "tenant_id": "",
            "query": "",
            "top_k": 0,
        },
    )

    assert response.status_code == 422

    mocked_dependencies["retriver_manager"].retrieve.assert_not_called()


def test_search_handles_value_error(
    client,
    mocked_dependencies,
):
    import semantic_search_eng.api.routes.search as search_module
    from semantic_search_eng.models.chunk import Chunk

    search_module._load_all_chunks = MagicMock(
        return_value=[
            Chunk(
                chunk_id="chunk_0",
                document_id="document_0000",
                tenant_id="conversation_001",
                chunk_index=0,
                text="Test chunk.",
                token_count=2,
            )
        ]
    )

    search_module._load_all_embeddings = MagicMock(
        return_value=[
            [1.0, 0.0, 0.0],
        ]
    )

    mocked_dependencies["retriver_manager"].retrieve.side_effect = ValueError("Invalid top_k")

    response = client.post(
        "/search",
        json={
            "tenant_id": "conversation_001",
            "query": "test",
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Invalid top_k")


def test_search_handles_missing_conversation(
    client,
    mocked_dependencies,
):
    import semantic_search_eng.api.routes.search as search_module

    search_module._load_all_chunks = MagicMock(
        side_effect=FileNotFoundError("Conversation not found")
    )

    response = client.post(
        "/search",
        json={
            "tenant_id": "missing",
            "query": "test",
            "top_k": 5,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("Conversation not found")
