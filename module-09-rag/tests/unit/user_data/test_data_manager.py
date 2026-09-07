from pathlib import Path

import pytest

from rag_app.user_data.data_manager import DataManager


def test_create_conversation_directory(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    conversation_path = manager.create_conversation_directory(str(tenant_id))

    assert conversation_path.exists()
    assert conversation_path.is_dir()

    assert (conversation_path / "documents").is_dir()


def test_save_and_get_document(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    content = "This is a test document."

    path = manager.save_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content=content,
    )

    assert path.exists()
    assert path.is_file()

    assert (
        path.read_text(
            encoding="utf-8",
        )
        == content
    )

    stored_content = manager.get_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
    )

    assert stored_content == content


def test_conversation_data_is_isolated(
    monkeypatch,
    test_settings,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    manager.save_document(
        tenant_id="conversation_001",
        document_id="document_001",
        content="Conversation one.",
    )

    manager.save_document(
        tenant_id="conversation_002",
        document_id="document_001",
        content="Conversation two.",
    )

    assert (
        manager.get_document(
            "conversation_001",
            "document_001",
        )
        == "Conversation one."
    )

    assert (
        manager.get_document(
            "conversation_002",
            "document_001",
        )
        == "Conversation two."
    )


def test_empty_tenant_id_is_rejected(
    monkeypatch,
    test_settings,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    with pytest.raises(
        ValueError,
        match="tenant_id cannot be empty",
    ):
        manager.create_conversation_directory("")


def test_create_conversation_directory_is_idempotent(
    monkeypatch,
    test_settings,
    tenant_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    first_path = manager.create_conversation_directory(str(tenant_id))

    second_path = manager.create_conversation_directory(str(tenant_id))

    assert first_path == second_path
    assert first_path.exists()
    assert (first_path / "documents").is_dir()


def test_save_document_overwrites_existing_document(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    manager.save_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content="Original content.",
    )

    manager.save_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content="Updated content.",
    )

    result = manager.get_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
    )

    assert result == "Updated content."


def test_save_empty_document(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    path = manager.save_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content="",
    )

    assert path.exists()

    assert (
        path.read_text(
            encoding="utf-8",
        )
        == ""
    )


def test_save_multiline_document(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    content = "First line.\nSecond line.\nThird line."

    path = manager.save_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content=content,
    )

    assert (
        path.read_text(
            encoding="utf-8",
        )
        == content
    )


def test_document_is_stored_under_expected_path(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    path = manager.save_document(
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content="Test content.",
    )

    expected_path = (
        Path(test_settings.data_directory)
        / str(tenant_id)
        / "documents"
        / f"{document_id}.txt"
    )

    assert path == expected_path


def test_get_missing_document_raises_error(
    monkeypatch,
    test_settings,
    tenant_id,
    document_id,
):
    monkeypatch.setattr(
        "rag_app.user_data.data_manager.get_settings",
        lambda: test_settings,
    )

    manager = DataManager()

    with pytest.raises(FileNotFoundError):
        manager.get_document(
            tenant_id=str(tenant_id),
            document_id=str(document_id),
        )
