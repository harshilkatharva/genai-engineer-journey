from uuid import uuid4
import pytest
from pydantic import ValidationError
from rag_app.models import UpsertRequest


def test_upsert_request_success():
    tid = uuid4()
    did = uuid4()
    req = UpsertRequest(
        tenant_id=tid,
        document_ids=[did],
        chunk_ids=["chunk_1"],
        updated_chunks=["text content"],
    )
    assert req.tenant_id == tid
    assert req.updated_chunks == ["text content"]


def test_upsert_request_length_mismatch_raises():
    with pytest.raises(ValidationError):
        UpsertRequest(
            tenant_id=uuid4(),
            document_ids=[uuid4(), uuid4()],
            chunk_ids=["chunk_1"],
            Updated_chunks=["text content"],
        )
