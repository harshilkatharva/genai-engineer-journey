from unittest.mock import MagicMock
from uuid import UUID

import pytest

from rag_app.chunking.chunking_manager import ChunkingManager


@pytest.fixture
def tenant_id():
    return UUID("bd7bc54b-27df-4f06-9d15-3de0e49cf103")


@pytest.fixture
def document_id():
    return UUID("bd7bc54b-27df-4f06-9d15-3de0e54cf103")


@pytest.fixture
def chunking_manager(monkeypatch, test_settings):
    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    mock_logger = MagicMock()

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: mock_logger,
    )

    manager = ChunkingManager()

    return manager, mock_logger


def test_chunk_documents_processes_multiple_documents(
    chunking_manager,
    tenant_id,
):
    manager, logger = chunking_manager

    document_id_1 = UUID("bd7bc54b-27df-4f06-9d15-3de0e54cf103")
    document_id_2 = UUID("bd7bc54b-27df-4f48-9d15-3de0e54cf103")

    documents = {
        document_id_1: "First document sentence.",
        document_id_2: "Second document sentence.",
    }

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents=documents,
        document_type=["text", "text"],
        metadata=[{}, {}],
    )

    assert len(chunks) == 2
    assert {chunk.document_id for chunk in chunks} == {
        document_id_1,
        document_id_2,
    }
    assert all(chunk.document_type == "text" for chunk in chunks)
    assert all(chunk.metadata == {} for chunk in chunks)

    logger.track.assert_called_once()


def test_chunk_documents_accepts_per_document_metadata(
    chunking_manager,
    tenant_id,
):
    manager, _ = chunking_manager

    document_id_1 = UUID("bd7bc54b-27df-4f06-9d15-3de0e54cf103")
    document_id_2 = UUID("bd7bc54b-27df-4f48-9d15-3de0e54cf103")

    documents = {
        document_id_1: "First document sentence.",
        document_id_2: "Second document sentence.",
    }

    document_type = ["pdf", "html"]
    metadata = [
        {"page": 1},
        {"url": "https://example.com"},
    ]

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents=documents,
        document_type=document_type,
        metadata=metadata,
    )

    assert len(chunks) == 2

    first = next(chunk for chunk in chunks if chunk.document_id == document_id_1)

    second = next(chunk for chunk in chunks if chunk.document_id == document_id_2)

    assert first.document_type == "pdf"
    assert first.metadata == {"page": 1}
    assert second.document_type == "html"
    assert second.metadata == {"url": "https://example.com"}


def test_chunk_documents_rejects_wrong_document_type_count(
    chunking_manager,
    tenant_id,
):
    manager, _ = chunking_manager

    documents = {
        UUID("bd7bc54b-27df-4f06-9d15-3de0e54cf103"): "First document.",
        UUID("bd7bc54b-27df-4f48-9d15-3de0e54cf103"): "Second document.",
    }

    with pytest.raises(ValueError, match="document_type count"):
        manager.chunk_documents(
            tenant_id=tenant_id,
            documents=documents,
            document_type=["pdf"],
            metadata=[{}, {}],
        )


def test_chunk_documents_rejects_wrong_metadata_count(
    chunking_manager,
    tenant_id,
):
    manager, _ = chunking_manager

    documents = {
        UUID("bd7bc54b-27df-4f06-9d15-3de0e54cf103"): "First document.",
        UUID("bd7bc54b-27df-4f48-9d15-3de0e54cf103"): "Second document.",
    }

    with pytest.raises(ValueError, match="metadata count"):
        manager.chunk_documents(
            tenant_id=tenant_id,
            documents=documents,
            document_type=["pdf", "html"],
            metadata=[{"page": 1}],
        )


def test_empty_document_returns_no_chunks(
    chunking_manager,
    tenant_id,
    document_id,
):
    manager, logger = chunking_manager

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: "   "},
        document_type=["text"],
        metadata=[{}],
    )

    assert chunks == []
    logger.track.assert_called_once()


def test_chunk_documents_records_tracker(
    chunking_manager,
    tenant_id,
    document_id,
):
    manager, logger = chunking_manager

    manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: "A simple test sentence."},
        document_type=["text"],
        metadata=[{}],
    )

    logger.track.assert_called_once()

    tracker = logger.track.call_args.args[0]

    assert tracker.tenant_id == str(tenant_id)
    assert tracker.document_count == 1
    assert tracker.total_chunks == 1
    assert tracker.chunking_strategy == "sentence"
    assert tracker.chunk_size == 500
    assert tracker.overlap == 50
    assert tracker.total_input_tokens > 0
    assert tracker.latency_ms >= 0


def test_large_sentence_is_not_split_mid_sentence(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    test_settings.chunk_size = 5
    test_settings.chunk_overlap = 0

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    text = "This is a very long sentence that contains many words and therefore exceeds the target."

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: text},
        document_type=["text"],
        metadata=[{}],
    )

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_overlap_keeps_previous_sentences(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    test_settings.chunk_size = 8
    test_settings.chunk_overlap = 4

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    text = "Alpha one two three. Beta one two three. Gamma one two three."

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: text},
        document_type=["text"],
        metadata=[{}],
    )

    assert len(chunks) >= 2

    assert any(
        sentence in chunks[1].text
        for sentence in (
            "Alpha one two three.",
            "Beta one two three.",
        )
    )


def test_chunk_preserves_sentence_boundaries(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    test_settings.chunk_size = 8
    test_settings.chunk_overlap = 0

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    text = "Alpha sentence is here. Beta sentence is here. Gamma sentence is here."

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: text},
        document_type=["text"],
        metadata=[{}],
    )

    assert len(chunks) == 3
    assert chunks[0].text == "Alpha sentence is here."
    assert chunks[1].text == "Beta sentence is here."
    assert chunks[2].text == "Gamma sentence is here."


def test_chunk_indexes_are_sequential(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    test_settings.chunk_size = 7
    test_settings.chunk_overlap = 0

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: ("One sentence here. Two sentence here. Three sentence here.")},
        document_type=["text"],
        metadata=[{}],
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    test_settings.chunk_size = 7
    test_settings.chunk_overlap = 0

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "rag_app.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents={document_id: ("One sentence. Two sentence. Three sentence.")},
        document_type=["text"],
        metadata=[{}],
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
