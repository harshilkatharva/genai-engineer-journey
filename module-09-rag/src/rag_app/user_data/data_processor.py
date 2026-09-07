from __future__ import annotations

from pathlib import Path
import uuid
from uuid import UUID

from rag_app.user_data.data_manager import DataManager


class DataProcessor:
    """
    Converts incoming document references into text documents
    that can be stored and processed by the semantic search pipeline.

    A document can be either:
    1. A path to an existing file.
    2. Raw document text.
    """

    def __init__(self) -> None:
        self.data_manager = DataManager()

    def process_document(
        self,
        tenant_id: UUID,
        document: str,
        document_id: UUID,
    ) -> UUID:
        """
        Process a document reference.

        If `document` points to an existing file, its contents are read.
        Otherwise the value is treated as raw document text.
        """
        if not document or not document.strip():
            raise ValueError("Document cannot be empty")

        content = self._resolve_document_content(document)

        self.data_manager.save_document(
            tenant_id=tenant_id,
            document_id=document_id,
            content=content,
        )

        return document_id

    def process_documents(
        self,
        tenant_id: UUID,
        documents: list[str],
    ) -> list[UUID]:
        self.data_manager.create_conversation_directory(tenant_id)

        document_ids: list[UUID] = []

        for document in documents:
            document_id = self.process_document(
                tenant_id=tenant_id,
                document=document,
                document_id=uuid.uuid4(),
            )

            document_ids.append(document_id)

        return document_ids

    @staticmethod
    def _resolve_document_content(
        document: str,
    ) -> str:
        """
        Determine whether the input is a file path or raw text.

        Newline-containing input is treated as document content.
        Short single-line strings are checked as file paths.

        OSError is caught because excessively long strings are not
        valid filesystem paths on typical Linux filesystems.
        """

        # Multiline input is overwhelmingly likely to be document content.
        if "\n" in document or "\r" in document:
            return document

        # Avoid trying to construct/check obviously huge path strings.
        if len(document) > 255:
            return document

        try:
            document_path = Path(document)

            if document_path.exists() and document_path.is_file():
                return document_path.read_text(encoding="utf-8")

        except OSError:
            # Treat invalid/too-long filesystem paths as raw content.
            return document

        return document
