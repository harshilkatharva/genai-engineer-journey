from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.services.index_services import IndexServiceManager


@pytest.fixture
def mock_dependencies():
    return {
        "data_processor": MagicMock(),
        "data_manager": MagicMock(),
        "chunking_manager": MagicMock(),
        "embedding_manager": MagicMock(),
        "index_db_manager": AsyncMock(),
    }


@pytest.fixture
def index_service_manager(monkeypatch, mock_dependencies):
    with (
        patch(
            "semantic_search_eng.services.index_services.DataProcessor",
            return_value=mock_dependencies["data_processor"],
        ),
        patch(
            "semantic_search_eng.services.index_services.DataManager",
            return_value=mock_dependencies["data_manager"],
        ),
        patch(
            "semantic_search_eng.services.index_services.ChunkingManager",
            return_value=mock_dependencies["chunking_manager"],
        ),
        patch(
            "semantic_search_eng.services.index_services.EmbeddingManager",
            return_value=mock_dependencies["embedding_manager"],
        ),
        patch(
            "semantic_search_eng.services.index_services.IndexDBManager",
            return_value=mock_dependencies["index_db_manager"],
        ),
    ):
        manager = IndexServiceManager()

    return manager, mock_dependencies


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk_0",
            document_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            document_type="pdf",
            metadata={"page": 1},
            chunk_index=0,
            text="First chunk.",
            token_count=2,
        ),
        Chunk(
            chunk_id="chunk_1",
            document_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
            document_type="pdf",
            metadata={"page": 1},
            chunk_index=1,
            text="Second chunk.",
            token_count=2,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


@pytest.mark.asyncio
async def test_index_orchestrates_document_processing(
    index_service_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, mocks = index_service_manager

    mocks["data_processor"].process_documents.return_value = [
        "doc_0",
        "doc_1",
    ]

    mocks["data_manager"].get_document.side_effect = [
        "Content of document 0",
        "Content of document 1",
    ]

    mocks["chunking_manager"].chunk_documents.return_value = sample_chunks

    mocks["embedding_manager"].embed_chunks.return_value = sample_embeddings

    mocks["index_db_manager"].store_index.return_value = None

    from semantic_search_eng.models.process_request import ProcessRequest

    request = ProcessRequest(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        documents=["Content 0", "Content 1"],
        documents_type=["type1", "type2"],
        meta_data=[{"key": "val1"}, {"key": "val2"}],
    )

    result = await manager.index(request)

    assert result["tenant_id"] == request.tenant_id
    assert result["document_count"] == 2
    assert result["chunk_count"] == len(sample_chunks)
    assert result["embedding_count"] == len(sample_embeddings)
    assert result["document_ids"] == ["doc_0", "doc_1"]


@pytest.mark.asyncio
async def test_index_calls_all_managers_in_sequence(
    index_service_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, mocks = index_service_manager

    mocks["data_processor"].process_documents.return_value = ["doc_0"]
    mocks["data_manager"].get_document.return_value = "Content"
    mocks["chunking_manager"].chunk_documents.return_value = sample_chunks
    mocks["embedding_manager"].embed_chunks.return_value = sample_embeddings

    from semantic_search_eng.models.process_request import ProcessRequest

    request = ProcessRequest(
        tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        documents=["Content"],
        documents_type=["type1"],
        meta_data=[{"key": "val"}],
    )

    await manager.index(request)

    # Verify calling sequence
    assert mocks["data_processor"].process_documents.called
    assert mocks["data_manager"].get_document.called
    assert mocks["chunking_manager"].chunk_documents.called
    assert mocks["embedding_manager"].embed_chunks.called
    assert mocks["index_db_manager"].store_index.called


@pytest.mark.asyncio
async def test_index_passes_correct_parameters(
    index_service_manager,
    sample_chunks,
    sample_embeddings,
):
    manager, mocks = index_service_manager

    mocks["data_processor"].process_documents.return_value = ["doc_0"]
    mocks["data_manager"].get_document.return_value = "Content"
    mocks["chunking_manager"].chunk_documents.return_value = sample_chunks
    mocks["embedding_manager"].embed_chunks.return_value = sample_embeddings

    from semantic_search_eng.models.process_request import ProcessRequest

    tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")
    request = ProcessRequest(
        tenant_id=tenant_id,
        documents=["Content"],
        documents_type=["pdf"],
        meta_data=[{"author": "test"}],
    )

    await manager.index(request)

    # Verify parameters
    assert mocks["data_processor"].process_documents.call_args[1]["tenant_id"] == tenant_id
    assert mocks["chunking_manager"].chunk_documents.call_args[1]["document_type"] == ["pdf"]
    assert mocks["chunking_manager"].chunk_documents.call_args[1]["metadata"] == [
        {"author": "test"}
    ]
    assert mocks["embedding_manager"].embed_chunks.call_args[1]["tenant_id"] == tenant_id
