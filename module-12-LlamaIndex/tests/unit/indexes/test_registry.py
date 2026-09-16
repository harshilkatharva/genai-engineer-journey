import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_app.indexes.registry import IndexFactory, IndexRegistry

from llama_index.core.schema import TextNode


def make_settings(tmp_path, postgres_url="postgresql://db"):
    return MagicMock(
        index_storage_dir=Path(tmp_path),
        postgres_url=postgres_url,
        vector_table_name="vectors",
        vector_schema_name="public",
        embedding_dimension=384,
    )


def test_build_creates_vector_and_summary_indexes(tmp_path) -> None:
    settings = make_settings(tmp_path)
    vector_store = MagicMock()
    vector_index = MagicMock()
    summary_index = MagicMock()
    nodes = [TextNode(text="test document")]

    with (
        patch("rag_app.indexes.registry.PGVectorStore", return_value=vector_store),
        patch("rag_app.indexes.registry.StorageContext.from_defaults") as storage_factory,
        patch("rag_app.indexes.registry.VectorStoreIndex", return_value=vector_index),
        patch("rag_app.indexes.registry.SummaryIndex", return_value=summary_index),
    ):
        registry = IndexFactory(settings).build(nodes)

    assert registry == IndexRegistry(vector=vector_index, summary=summary_index)
    assert storage_factory.call_count == 2
    storage_factory.return_value.persist.assert_called_once_with(persist_dir=Path(tmp_path))


def test_vector_store_requires_postgres_url(tmp_path) -> None:
    factory = IndexFactory(make_settings(tmp_path, postgres_url=""))

    with pytest.raises(RuntimeError, match="POSTGRES_URL"):
        factory._vector_store()


def test_manifest_is_written_and_contains_counts(tmp_path) -> None:
    factory = IndexFactory(make_settings(tmp_path))
    factory.persist_manifest(source_dir=Path("/corpus"), documents=4, nodes=18)

    manifest = json.loads(factory.manifest_path.read_text())

    assert manifest["source_dir"] == "/corpus"
    assert manifest["documents"] == 4
    assert manifest["nodes"] == 18


def test_restore_returns_persisted_indexes_and_metadata(tmp_path) -> None:
    factory = IndexFactory(make_settings(tmp_path))
    factory.persist_manifest(source_dir=Path("/corpus"), documents=2, nodes=5)
    summary_index = MagicMock()
    vector_index = MagicMock()

    with (
        patch("rag_app.indexes.registry.StorageContext.from_defaults"),
        patch("rag_app.indexes.registry.load_index_from_storage", return_value=summary_index),
        patch.object(factory, "_vector_store", return_value=MagicMock()),
        patch(
            "rag_app.indexes.registry.VectorStoreIndex.from_vector_store",
            return_value=vector_index,
        ),
    ):
        restored = factory.restore()

    assert restored is not None
    registry, metadata = restored
    assert registry.vector is vector_index
    assert registry.summary is summary_index
    assert metadata["documents"] == 2


def test_restore_returns_none_without_manifest(tmp_path) -> None:
    assert IndexFactory(make_settings(tmp_path)).restore() is None
