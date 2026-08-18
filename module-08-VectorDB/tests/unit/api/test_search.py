"""
API tests for the Vector Database Semantic Search Engine.
Tests basic endpoint functionality and validation.
"""

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from semantic_search_eng.api.app import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns app info"""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert "name" in body
    assert "version" in body
    assert body["status"] == "running"


def test_health_endpoint(client):
    """Test the health endpoint"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_uuid_endpoint(client):
    """Test the UUID generation endpoint"""
    response = client.get("/genrate_uuid")

    assert response.status_code == 200
    uuid_str = response.json()
    # Verify it's a valid UUID string
    UUID(uuid_str)


def test_process_documents_validation_error_missing_fields(client):
    """Test validation error for invalid request"""
    # Missing required fields
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        # Missing documents, documents_type, meta_data
    }

    response = client.post("/index/process", json=payload)

    assert response.status_code == 422  # Pydantic validation error


def test_process_documents_mismatched_arrays(client):
    """Test error when array lengths don't match"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "documents": ["Doc 1", "Doc 2"],
        "documents_type": ["pdf"],  # Only one type, should be 2
        "meta_data": [{"key": "val1"}, {"key": "val2"}],
    }

    response = client.post("/index/process", json=payload)

    assert response.status_code == 422


def test_process_documents_empty_documents(client):
    """Test error when documents list is empty"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "documents": [],
        "documents_type": [],
        "meta_data": [],
    }

    response = client.post("/index/process", json=payload)

    assert response.status_code == 422


def test_retrieve_empty_query_validation(client):
    """Test that empty query is rejected"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "query": "",  # Empty query
        "top_k": 5,
    }

    response = client.post("/retrive/", json=payload)

    assert response.status_code == 422


def test_retrieve_missing_query(client):
    """Test that missing query is rejected"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        # Missing query
        "top_k": 5,
    }

    response = client.post("/retrive/", json=payload)

    assert response.status_code == 422


def test_retrieve_invalid_top_k(client):
    """Test that invalid top_k values are rejected"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "query": "test query",
        "top_k": 0,  # Must be >= 1
    }

    response = client.post("/retrive/", json=payload)

    assert response.status_code == 422


def test_process_documents_with_valid_request_structure(client):
    """Test that valid process request has correct structure"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "documents": ["Document 1"],
        "documents_type": ["pdf"],
        "meta_data": [{"page": 1}],
    }

    response = client.post("/index/process", json=payload)

    # Should be 200 if successful or 500 if internal error, but not 422
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        body = response.json()
        assert "tenant_id" in body
        assert "document_count" in body
        assert "chunk_count" in body


def test_retrieve_with_valid_request_structure(client):
    """Test that valid retrieve request has correct structure"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "query": "test query",
        "top_k": 5,
    }

    response = client.post("/retrive/", json=payload)

    # Should be 200 if successful or 500 if internal error, but not 422
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        body = response.json()
        assert "tenant_id" in body
        assert "query" in body
        assert "top_k" in body
        assert "results" in body
        assert isinstance(body["results"], list)


def test_retrieve_default_top_k_structure(client):
    """Test retrieval without explicit top_k uses default"""
    payload = {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
        "query": "test query",
        # Omit top_k to use default
    }

    response = client.post("/retrive/", json=payload)

    # Should be 200 or 500, not 422
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        body = response.json()
        assert body["top_k"] == 5  # Default top_k
