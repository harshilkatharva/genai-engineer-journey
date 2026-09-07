from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_app.core.settings import get_settings
from rag_app.models.retrive.retrive_request import RetriveRequest

settings = get_settings()


def test_retrive_request_with_valid_data():
    tenant_id = uuid4()

    request = RetriveRequest(
        tenant_id=tenant_id,
        query="What is the refund policy?",
        top_k_candidate=5,
        top_k_re_ranker=3,
    )

    assert request.tenant_id == tenant_id
    assert request.query == "What is the refund policy?"
    assert request.top_k_candidate == 5
    assert request.top_k_re_ranker == 3


def test_retrive_request_uses_default_values():
    tenant_id = uuid4()

    request = RetriveRequest(
        tenant_id=tenant_id,
        query="What is the refund policy?",
    )

    assert request.tenant_id == tenant_id
    assert request.query == "What is the refund policy?"

    # Values come from settings
    assert request.top_k_candidate == settings.canidate_default_top_k
    assert request.top_k_re_ranker == settings.re_ranker_default_top_k


def test_retrive_request_requires_query():
    tenant_id = uuid4()

    with pytest.raises(ValidationError, match="query"):
        RetriveRequest(
            tenant_id=tenant_id,
        )


def test_retrive_request_rejects_invalid_tenant_id():
    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id="not-a-uuid",
            query="test query",
        )


def test_retrive_request_accepts_single_query():
    tenant_id = uuid4()

    query = "What is the refund policy?"

    request = RetriveRequest(
        tenant_id=tenant_id,
        query=query,
    )

    assert request.query == query


def test_top_k_candidate_must_be_at_least_one():
    tenant_id = uuid4()

    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id=tenant_id,
            query="test query",
            top_k_candidate=0,
        )


def test_top_k_re_ranker_must_be_at_least_one():
    tenant_id = uuid4()

    with pytest.raises(ValidationError):
        RetriveRequest(
            tenant_id=tenant_id,
            query="test query",
            top_k_re_ranker=0,
        )
