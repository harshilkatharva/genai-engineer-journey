import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_app.indexes.registry import IndexFactory


def make_settings(tmp_path):
    return MagicMock(
        index_storage_dir=Path(tmp_path),
        postgres_url="postgresql://db",
        vector_table_name="vectors",
        vector_schema_name="public",
        embedding_dimension=384,
    )


def test_manifest_persistence_writes_expected_storage_metadata(tmp_path) -> None:
    factory = IndexFactory(make_settings(tmp_path))

    factory.persist_manifest(source_dir=Path("/corpus"), documents=3, nodes=10)

    manifest = json.loads(factory.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_dir"] == "/corpus"
    assert manifest["documents"] == 3
    assert manifest["nodes"] == 10
    assert manifest["vector_table_name"] == "vectors"
    assert manifest["vector_schema_name"] == "public"


def test_restore_reads_manifest_and_rebuilds_index_objects(tmp_path) -> None:
    factory = IndexFactory(make_settings(tmp_path))
    factory.persist_manifest(source_dir=Path("/docs"), documents=5, nodes=11)
    summary_index = MagicMock()
    vector_index = MagicMock()

    with (
        patch("rag_app.indexes.registry.StorageContext.from_defaults"),
        patch("rag_app.indexes.registry.load_index_from_storage", return_value=summary_index),
        patch.object(factory, "_vector_store", return_value=MagicMock()),
        patch(
            "rag_app.indexes.registry.VectorStoreIndex.from_vector_store", return_value=vector_index
        ),
    ):
        restored = factory.restore()

    assert restored is not None
    registry, metadata = restored
    assert registry.vector is vector_index
    assert registry.summary is summary_index
    assert metadata["documents"] == 5
    assert metadata["nodes"] == 11
