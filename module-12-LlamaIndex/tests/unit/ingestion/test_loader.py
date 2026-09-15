from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_app.ingestion.loader import DocumentLoader


def test_load_reads_documents_and_chunks_them(tmp_path) -> None:
    settings = MagicMock(chunk_size=100, chunk_overlap=20)
    reader = MagicMock()
    document = MagicMock()
    nodes = [MagicMock()]
    reader.load_data.return_value = [document]

    with (
        patch("rag_app.ingestion.loader.SimpleDirectoryReader", return_value=reader) as reader_cls,
        patch("rag_app.ingestion.loader.SentenceSplitter") as splitter_cls,
    ):
        splitter_cls.return_value.get_nodes_from_documents.return_value = nodes
        corpus = DocumentLoader(settings).load(tmp_path)

    reader_cls.assert_called_once_with(
        input_dir=Path(tmp_path).resolve(),
        recursive=True,
        exclude_hidden=True,
        filename_as_id=True,
    )
    splitter_cls.assert_called_once_with(chunk_size=100, chunk_overlap=20)
    assert corpus.source_dir == Path(tmp_path).resolve()
    assert corpus.documents == [document]
    assert corpus.nodes == nodes


def test_load_rejects_missing_directory(tmp_path) -> None:
    settings = MagicMock()

    with pytest.raises(FileNotFoundError, match="Data directory does not exist"):
        DocumentLoader(settings).load(tmp_path / "missing")


def test_load_rejects_empty_directory(tmp_path) -> None:
    reader = MagicMock()
    reader.load_data.return_value = []
    settings = MagicMock()

    with (
        patch("rag_app.ingestion.loader.SimpleDirectoryReader", return_value=reader),
        pytest.raises(ValueError, match="No supported documents"),
    ):
        DocumentLoader(settings).load(tmp_path)
