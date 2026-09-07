from pathlib import Path
from uuid import UUID

import pytest

from rag_app.user_data.data_processor import DataProcessor


def test_process_raw_document(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_processor.DataManager",
        lambda: __import__(
            "rag_app.user_data.data_manager",
            fromlist=["DataManager"],
        ).DataManager(),
    )

    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    processor = DataProcessor()

    document_id = processor.process_document(
        tenant_id=tenant_id,
        document="This is raw document content.",
        document_id="document_0000",
    )

    assert document_id == "document_0000"

    stored_path = (
        Path(test_settings.data_directory)
        / str(tenant_id)
        / "documents"
        / "document_0000.txt"
    )

    assert stored_path.exists()
    assert stored_path.read_text(encoding="utf-8") == "This is raw document content."


def test_process_file_document(
    monkeypatch,
    test_settings,
    tenant_id,
    tmp_path,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    source_file = tmp_path / "notes.txt"
    source_file.write_text(
        "This content came from a file.",
        encoding="utf-8",
    )

    processor = DataProcessor()

    document_id = processor.process_document(
        tenant_id=tenant_id,
        document=str(source_file),
        document_id="document_from_file",
    )

    assert document_id == "document_from_file"

    stored_path = (
        Path(test_settings.data_directory)
        / str(tenant_id)
        / "documents"
        / "document_from_file.txt"
    )

    assert stored_path.exists()
    assert stored_path.read_text(encoding="utf-8") == "This content came from a file."


def test_process_documents(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    processor = DataProcessor()

    document_ids = processor.process_documents(
        tenant_id=tenant_id,
        documents=[
            "First document.",
            "Second document.",
            "Third document.",
        ],
    )

    assert len(document_ids) == 3

    assert all(isinstance(document_id, UUID) for document_id in document_ids)

    assert len(set(document_ids)) == 3


def test_empty_document_is_rejected(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    processor = DataProcessor()

    with pytest.raises(
        ValueError,
        match="Document cannot be empty",
    ):
        processor.process_document(
            tenant_id=tenant_id,
            document="   ",
            document_id="document_0000",
        )


def test_process_multiline_raw_document(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    processor = DataProcessor()

    content = "First line.\nSecond line.\nThird line."

    document_id = processor.process_document(
        tenant_id=tenant_id,
        document=content,
        document_id="multiline_document",
    )

    assert document_id == "multiline_document"

    stored_path = (
        Path(test_settings.data_directory)
        / str(tenant_id)
        / "documents"
        / "multiline_document.txt"
    )

    assert (
        stored_path.read_text(
            encoding="utf-8",
        )
        == content
    )


def test_process_long_raw_document(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    processor = DataProcessor()

    content = "a" * 256

    document_id = processor.process_document(
        tenant_id=tenant_id,
        document=content,
        document_id="large_document",
    )

    assert document_id == "large_document"

    stored_path = (
        Path(test_settings.data_directory)
        / str(tenant_id)
        / "documents"
        / "large_document.txt"
    )

    assert stored_path.read_text(encoding="utf-8") == content


def test_conversation_directories_are_isolated(
    monkeypatch,
    test_settings,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    processor = DataProcessor()

    tenant_one = UUID("11111111-1111-1111-1111-111111111111")
    tenant_two = UUID("22222222-2222-2222-2222-222222222222")

    processor.process_document(
        tenant_id=tenant_one,
        document="Tenant one document.",
        document_id="document_001",
    )

    processor.process_document(
        tenant_id=tenant_two,
        document="Tenant two document.",
        document_id="document_001",
    )

    path_one = (
        Path(test_settings.data_directory)
        / str(tenant_one)
        / "documents"
        / "document_001.txt"
    )

    path_two = (
        Path(test_settings.data_directory)
        / str(tenant_two)
        / "documents"
        / "document_001.txt"
    )

    assert path_one.read_text(encoding="utf-8") == ("Tenant one document.")

    assert path_two.read_text(encoding="utf-8") == ("Tenant two document.")
