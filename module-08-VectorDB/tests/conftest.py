from types import SimpleNamespace

import pytest


@pytest.fixture
def test_settings(tmp_path):
    """
    Shared settings for filesystem-based unit tests.

    Every test gets its own temporary data directory.
    """
    return SimpleNamespace(
        data_directory=str(tmp_path / "data"),
        log_directory=str(tmp_path / "logs"),
        chunking_strategy="sentence",
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        embedding_batch_size=100,
        embedding_cost_per_million_tokens=0.0,
        default_top_k=5,
        max_top_k=100,
    )


@pytest.fixture
def tenant_id() -> str:
    return "conversation_test_001"


@pytest.fixture
def document_id() -> str:
    return "document_0000"


@pytest.fixture
def sample_document() -> str:
    return (
        "Semantic search finds information by meaning. "
        "It converts text into numerical embeddings. "
        "Similar meanings produce similar vectors. "
        "A retriever compares the query vector with document vectors. "
        "The highest scoring chunks are returned to the user."
    )
