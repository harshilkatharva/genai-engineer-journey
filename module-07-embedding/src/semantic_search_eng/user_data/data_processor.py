from __future__ import annotations

from pathlib import Path

from semantic_search_eng.user_data.data_manager import DataManager


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
        conversation_id: str,
        document: str,
        document_id: str | None = None,
    ) -> str:
        """
        Process a document reference.

        If `document` points to an existing file, its contents are read.
        Otherwise the value is treated as raw document text.
        """

        if not document or not document.strip():
            raise ValueError("Document cannot be empty")

        resolved_document_id = document_id or self._generate_document_id(document)

        content = self._resolve_document_content(document)

        self.data_manager.save_document(
            conversation_id=conversation_id,
            document_id=resolved_document_id,
            content=content,
        )

        return resolved_document_id

    def process_documents(
        self,
        conversation_id: str,
        documents: list[str],
    ) -> list[str]:
        self.data_manager.create_conversation_directory(conversation_id)

        document_ids: list[str] = []

        for index, document in enumerate(documents):
            document_id = self.process_document(
                conversation_id=conversation_id,
                document=document,
                document_id=f"document_{index:04d}",
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

    @staticmethod
    def _generate_document_id(
        document: str,
    ) -> str:
        """
        Generate a sensible document ID.

        For a real file path, use its filename stem.
        For raw text, use a generic document ID.
        """

        # Raw multiline content should not become a filesystem path.
        if "\n" in document or "\r" in document:
            return "document"

        if len(document) > 255:
            return "document"

        try:
            document_path = Path(document)

            if document_path.exists() and document_path.is_file():
                if document_path.stem:
                    return document_path.stem

        except OSError:
            pass

        return "document"
