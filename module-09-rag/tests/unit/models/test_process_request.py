from uuid import UUID

import pytest
from pydantic import ValidationError

from rag_app.models.process_request import ProcessRequest


def test_process_request_creation() -> None:
    tenant_id = "550e8400-e29b-41d4-a716-446655440000"
    request = ProcessRequest(
        tenant_id=tenant_id,
        documents=["document one", "document two"],
        documents_type=["type1", "type2"],
        meta_data=[{"key": "value1"}, {"key": "value2"}],
    )

    assert request.tenant_id == UUID(tenant_id)
    assert request.documents == [
        "document one",
        "document two",
    ]
    assert request.documents_type == ["type1", "type2"]
    assert request.meta_data == [{"key": "value1"}, {"key": "value2"}]


def test_process_request_rejects_empty_documents() -> None:
    with pytest.raises(ValidationError):
        ProcessRequest(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            documents=[],
            documents_type=[],
            meta_data=[],
        )


def test_process_request_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        ValidationError, match="documents, documents_type, and meta_data must have the same length"
    ):
        ProcessRequest(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            documents=["document one", "document two"],
            documents_type=["type1"],
            meta_data=[{"key": "value1"}, {"key": "value2"}],
        )


def test_process_request_invalid_tenant_id() -> None:
    with pytest.raises(ValidationError):
        ProcessRequest(
            tenant_id="not-a-uuid",
            documents=["document"],
            documents_type=["type1"],
            meta_data=[{"key": "value"}],
        )
