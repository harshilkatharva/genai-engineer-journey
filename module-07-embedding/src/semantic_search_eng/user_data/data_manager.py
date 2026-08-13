from pathlib import Path
import json
from typing import Any

from semantic_search_eng.config import get_settings


class DataManager:
    """
    Handles all persistent storage for conversations.

    Storage layout:

        user_data/data/
            {conversation_id}/
                documents/
                chunks/
                embeddings/
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_path = Path(self.settings.data_directory)

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def create_conversation_directory(
        self,
        conversation_id: str,
    ) -> Path:
        conversation_path = self._conversation_path(conversation_id)

        (conversation_path / "documents").mkdir(
            parents=True,
            exist_ok=True,
        )

        (conversation_path / "chunks").mkdir(
            parents=True,
            exist_ok=True,
        )

        (conversation_path / "embeddings").mkdir(
            parents=True,
            exist_ok=True,
        )

        return conversation_path

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def save_document(
        self,
        conversation_id: str,
        document_id: str,
        content: str,
    ) -> Path:
        conversation_path = self.create_conversation_directory(conversation_id)

        document_path = conversation_path / "documents" / f"{document_id}.txt"

        document_path.write_text(
            content,
            encoding="utf-8",
        )

        return document_path

    def get_document(
        self,
        conversation_id: str,
        document_id: str,
    ) -> str:
        document_path = (
            self._conversation_path(conversation_id) / "documents" / f"{document_id}.txt"
        )

        return document_path.read_text(
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def save_chunks(
        self,
        conversation_id: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> Path:
        conversation_path = self.create_conversation_directory(conversation_id)

        chunk_path = conversation_path / "chunks" / f"{document_id}.json"

        chunk_path.write_text(
            json.dumps(
                chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return chunk_path

    def get_chunks(
        self,
        conversation_id: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        chunk_path = self._conversation_path(conversation_id) / "chunks" / f"{document_id}.json"

        if not chunk_path.exists():
            return []

        return json.loads(
            chunk_path.read_text(
                encoding="utf-8",
            )
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embeddings(
        self,
        conversation_id: str,
        document_id: str,
        embeddings: list[list[float]],
    ) -> Path:
        conversation_path = self.create_conversation_directory(conversation_id)

        embedding_path = conversation_path / "embeddings" / f"{document_id}.json"

        embedding_path.write_text(
            json.dumps(
                embeddings,
            ),
            encoding="utf-8",
        )

        return embedding_path

    def get_embeddings(
        self,
        conversation_id: str,
        document_id: str,
    ) -> list[list[float]]:
        embedding_path = (
            self._conversation_path(conversation_id) / "embeddings" / f"{document_id}.json"
        )

        if not embedding_path.exists():
            return []

        return json.loads(
            embedding_path.read_text(
                encoding="utf-8",
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conversation_path(
        self,
        conversation_id: str,
    ) -> Path:
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")

        return self.base_path / conversation_id
