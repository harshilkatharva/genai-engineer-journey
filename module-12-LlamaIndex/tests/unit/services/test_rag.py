from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_app.schemas.rag import QueryResponse
from rag_app.services.rag import RAGService


def make_settings(tmp_path):
    return MagicMock(
        postgres_url="",
        index_storage_dir=Path(tmp_path),
    )


def test_service_starts_unready_without_persisted_state(tmp_path) -> None:
    service = RAGService(make_settings(tmp_path))

    assert service.ready is False
    assert service.document_count == 0


@pytest.mark.asyncio
async def test_ingest_configures_models_builds_indexes_and_persists_manifest(tmp_path) -> None:
    settings = make_settings(tmp_path)
    service = RAGService(settings)
    document = MagicMock()
    node = MagicMock()
    corpus = MagicMock(source_dir=Path("/corpus"), documents=[document], nodes=[node])
    ingest_response = service.ingest

    with (
        patch.object(service.loader, "load", return_value=corpus),
        patch.object(service.index_factory, "build", return_value=MagicMock()),
        patch.object(service.index_factory, "persist_manifest") as persist_manifest,
        patch("rag_app.services.rag.ModelProvider") as provider_cls,
    ):
        result = await ingest_response("/corpus")

    provider_cls.return_value.configure.assert_called_once()
    persist_manifest.assert_called_once_with(
        source_dir=Path("/corpus"),
        documents=1,
        nodes=1,
    )
    assert result.source_dir == "/corpus"
    assert result.documents == 1
    assert service.ready is True


@pytest.mark.asyncio
async def test_query_routes_to_vector_and_extracts_sources(tmp_path) -> None:
    service = RAGService(make_settings(tmp_path))
    service.indexes = MagicMock()
    source_node = MagicMock()
    source_node.node.metadata = {"file_name": "report.txt"}
    source_node.node.get_content.return_value = "evidence" * 100
    response = MagicMock(source_nodes=[source_node])
    vector_engine = MagicMock()
    vector_engine.aquery = AsyncMock(return_value=response)

    with patch("rag_app.services.rag.RetrievalEngineFactory") as engines_cls:
        engines_cls.return_value.vector.return_value = vector_engine
        result = await service.query("What happened?", route="vector")

    vector_engine.aquery.assert_awaited_once_with("What happened?")
    assert isinstance(result, QueryResponse)
    assert result.route == "vector"
    assert result.sources[0].source == "report.txt"
    assert len(result.sources[0].snippet) == 300


@pytest.mark.asyncio
async def test_query_rejects_unready_service(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Run POST /ingest"):
        await RAGService(make_settings(tmp_path)).query("question")


@pytest.mark.asyncio
async def test_query_uses_router_when_route_not_provided(tmp_path) -> None:
    service = RAGService(make_settings(tmp_path))
    service.indexes = MagicMock()

    source_node = MagicMock()
    source_node.node.metadata = {"file_name": "summary.txt"}
    source_node.node.get_content.return_value = "summary evidence"
    response = MagicMock(source_nodes=[source_node])
    summary_engine = MagicMock()
    summary_engine.aquery = AsyncMock(return_value=response)

    with (
        patch.object(service.router, "classify", return_value="summary") as classify_mock,
        patch("rag_app.services.rag.RetrievalEngineFactory") as engines_cls,
    ):
        engines_cls.return_value.summary.return_value = summary_engine
        result = await service.query("Give me an overview")

    classify_mock.assert_called_once_with("Give me an overview")
    summary_engine.aquery.assert_awaited_once_with("Give me an overview")
    assert result.route == "summary"
    assert result.sources[0].source == "summary.txt"


@pytest.mark.asyncio
async def test_query_uses_file_path_fallback_when_file_name_missing(tmp_path) -> None:
    service = RAGService(make_settings(tmp_path))
    service.indexes = MagicMock()

    source_node = MagicMock()
    source_node.node.metadata = {"file_path": "/tmp/doc.txt"}
    source_node.node.get_content.return_value = "fallback stuff"
    response = MagicMock(source_nodes=[source_node])
    vector_engine = MagicMock()
    vector_engine.aquery = AsyncMock(return_value=response)

    with patch("rag_app.services.rag.RetrievalEngineFactory") as engines_cls:
        engines_cls.return_value.vector.return_value = vector_engine
        result = await service.query("What is it?", route="vector")

    assert result.sources[0].source == "/tmp/doc.txt"
    assert result.sources[0].snippet == "fallback stuff"


@pytest.mark.asyncio
async def test_query_handles_empty_source_nodes(tmp_path) -> None:
    service = RAGService(make_settings(tmp_path))
    service.indexes = MagicMock()
    response = MagicMock(source_nodes=[])
    vector_engine = MagicMock()
    vector_engine.aquery = AsyncMock(return_value=response)

    with patch("rag_app.services.rag.RetrievalEngineFactory") as engines_cls:
        engines_cls.return_value.vector.return_value = vector_engine
        result = await service.query("Empty result", route="vector")

    assert result.sources == []
    assert result.answer == str(response)


def test_restore_persisted_indexes_updates_service_state(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.postgres_url = "postgresql://example/db"
    service = RAGService(settings)
    manifest_path = settings.index_storage_dir / "manifest.json"
    manifest_path.write_text("{}")

    restored_index = MagicMock()
    restored_metadata = {"documents": "7", "nodes": "11"}

    with (
        patch.object(
            service.index_factory, "restore", return_value=(restored_index, restored_metadata)
        ),
        patch("rag_app.services.rag.ModelProvider") as provider_cls,
    ):
        service._restore_persisted_indexes()

    provider_cls.return_value.configure.assert_called_once()
    assert service.indexes is restored_index
    assert service.document_count == 7
    assert service._node_count == 11
    assert service.ready is True


@pytest.mark.asyncio
async def test_async_dependency_can_be_injected_for_async_callers() -> None:
    dependency = AsyncMock(return_value="ready")

    result = await dependency()

    dependency.assert_awaited_once_with()
    assert result == "ready"
