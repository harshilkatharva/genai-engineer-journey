from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from rag_app.models import UpsertRequest
from rag_app.services.upsert_services import UpsertServiceManager


@pytest.fixture
def upsert_service():
    service = UpsertServiceManager()

    service.data_manager = MagicMock()
    service.data_processor = MagicMock()
    service.embedding_manager = MagicMock()
    service.embedding_manager.embed_chunks = AsyncMock()

    service.upsert_db_manager = MagicMock()
    service.upsert_db_manager.upsert_chunks = AsyncMock()

    service.upsert_db_manager.upsert_chunks = AsyncMock()

    return service


@pytest.mark.asyncio
async def test_upsert_chunks_success(upsert_service):
    tenant_id = uuid4()
    document_id_1 = uuid4()
    document_id_2 = uuid4()

    request = UpsertRequest(
        tenant_id=tenant_id,
        document_ids=[
            document_id_1,
            document_id_2,
        ],
        chunk_ids=[
            "chunk_1",
            "chunk_2",
        ],
        updated_chunks=[
            "updated text 1",
            "updated text 2",
        ],
    )

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    expected_result = ["chunk_1", "chunk_2"]

    upsert_service.embedding_manager.embed_chunks.return_value = embeddings
    upsert_service.upsert_db_manager.upsert_chunks.return_value = expected_result

    result = await upsert_service.upsert_chunks(request)

    assert result == expected_result

    upsert_service.embedding_manager.embed_chunks.assert_called_once_with(
        tenant_id=tenant_id,
        chunks=request.updated_chunks,
    )

    upsert_service.upsert_db_manager.upsert_chunks.assert_awaited_once_with(
        tenant_id=tenant_id,
        updated_chunks=request.updated_chunks,
        updated_embeddings=embeddings,
        document_ids=[
            document_id_1,
            document_id_2,
        ],
        chunk_ids=[
            "chunk_1",
            "chunk_2",
        ],
    )


@pytest.mark.asyncio
async def test_upsert_chunks_embedding_error(upsert_service):
    tenant_id = uuid4()

    request = UpsertRequest(
        tenant_id=tenant_id,
        document_ids=[uuid4()],
        chunk_ids=["chunk_1"],
        updated_chunks=["updated text"],
    )

    upsert_service.embedding_manager.embed_chunks.side_effect = RuntimeError(
        "Embedding service unavailable"
    )

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        await upsert_service.upsert_chunks(request)

    upsert_service.upsert_db_manager.upsert_chunks.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_chunks_db_error(upsert_service):
    tenant_id = uuid4()

    request = UpsertRequest(
        tenant_id=tenant_id,
        document_ids=[uuid4()],
        chunk_ids=["chunk_1"],
        updated_chunks=["updated text"],
    )

    embeddings = [[0.1, 0.2, 0.3]]

    upsert_service.embedding_manager.embed_chunks.return_value = embeddings

    upsert_service.upsert_db_manager.upsert_chunks.side_effect = RuntimeError(
        "Database update failed"
    )

    with pytest.raises(RuntimeError, match="Database update failed"):
        await upsert_service.upsert_chunks(request)

    upsert_service.embedding_manager.embed_chunks.assert_called_once()
    upsert_service.upsert_db_manager.upsert_chunks.assert_awaited_once()
