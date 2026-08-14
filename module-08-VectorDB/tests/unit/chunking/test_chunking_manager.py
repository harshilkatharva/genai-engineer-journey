from unittest.mock import MagicMock

import pytest

from semantic_search_eng.chunking.chunking_manager import (
    ChunkingManager,
)


@pytest.fixture
def chunking_manager(
    monkeypatch,
    test_settings,
):
    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    mock_logger = MagicMock()

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: mock_logger,
    )

    manager = ChunkingManager()

    return manager, mock_logger


def test_chunk_document_returns_chunks(
    chunking_manager,
    tenant_id,
    document_id,
):
    manager, _ = chunking_manager

    text = (
        "The first sentence explains semantic search. "
        "The second sentence explains embeddings. "
        "The third sentence explains retrieval."
    )

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text=text,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.chunk_id == ("document_0000_chunk_0000")
    assert chunk.document_id == document_id
    assert chunk.tenant_id == tenant_id
    assert chunk.chunk_index == 0
    assert chunk.text == text
    assert chunk.token_count > 0


def test_chunk_preserves_sentence_boundaries(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    # Force a very small target so multiple chunks are created.
    test_settings.chunk_size = 8
    test_settings.chunk_overlap = 0

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    mock_logger = MagicMock()

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: mock_logger,
    )

    manager = ChunkingManager()

    text = "Alpha sentence is here. Beta sentence is here. Gamma sentence is here."

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text=text,
    )

    assert len(chunks) == 3

    assert chunks[0].text == ("Alpha sentence is here.")
    assert chunks[1].text == ("Beta sentence is here.")
    assert chunks[2].text == ("Gamma sentence is here.")


def test_chunk_indexes_are_sequential(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    test_settings.chunk_size = 7
    test_settings.chunk_overlap = 0

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text=("One sentence here. Two sentence here. Three sentence here."),
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
        "semantic_search_eng.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text=("One sentence. Two sentence. Three sentence."),
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_document_returns_no_chunks(
    chunking_manager,
    tenant_id,
    document_id,
):
    manager, logger = chunking_manager

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text="   ",
    )

    assert chunks == []

    logger.track.assert_not_called()


def test_chunk_documents_processes_multiple_documents(
    chunking_manager,
    tenant_id,
):
    manager, logger = chunking_manager

    documents = {
        "document_0000": "First document sentence.",
        "document_0001": "Second document sentence.",
    }

    chunks = manager.chunk_documents(
        tenant_id=tenant_id,
        documents=documents,
    )

    assert len(chunks) == 2

    assert {chunk.document_id for chunk in chunks} == {
        "document_0000",
        "document_0001",
    }

    logger.track.assert_called_once()


def test_chunking_tracker_is_recorded(
    chunking_manager,
    tenant_id,
    document_id,
):
    manager, logger = chunking_manager

    manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text="A simple test sentence.",
    )

    logger.track.assert_called_once()

    tracker = logger.track.call_args.args[0]

    assert tracker.tenant_id == tenant_id
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
        "semantic_search_eng.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    text = "This is a very long sentence that contains many words and therefore exceeds the target."

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text=text,
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
        "semantic_search_eng.chunking.chunking_manager.get_settings",
        lambda: test_settings,
    )

    monkeypatch.setattr(
        "semantic_search_eng.chunking.chunking_manager.ChunkingTrackerLogger",
        lambda: MagicMock(),
    )

    manager = ChunkingManager()

    text = "Alpha one two three. Beta one two three. Gamma one two three."

    chunks = manager.chunk_document(
        tenant_id=tenant_id,
        document_id=document_id,
        text=text,
    )

    assert len(chunks) >= 2

    # Because overlap is sentence-based, the previous chunk's trailing
    # sentence(s) should appear at the beginning of the next chunk.
    assert "Beta one two three." in chunks[1].text or "Alpha one two three." in chunks[1].text
