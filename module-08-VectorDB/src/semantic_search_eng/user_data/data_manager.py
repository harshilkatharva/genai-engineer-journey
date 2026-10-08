import json
from pathlib import Path
from uuid import UUID

from semantic_search_eng.config import get_settings


class DataManager:
    """
    Handles all persistent storage for conversations.

    Storage layout:

        user_data/data/
            {tenant_id}/
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
        tenant_id: UUID,
    ) -> Path:
        conversation_path = self._conversation_path(tenant_id)

        (conversation_path / "documents").mkdir(
            parents=True,
            exist_ok=True,
        )

        return conversation_path

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def save_document(
        self,
        tenant_id: UUID,
        document_id: UUID,
        content: str,
    ) -> Path:
        conversation_path = self.create_conversation_directory(tenant_id)

        document_path = conversation_path / "documents" / f"{document_id}.txt"

        document_path.write_text(
            content,
            encoding="utf-8",
        )

        return document_path

    def get_document(
        self,
        tenant_id: UUID,
        document_id: str,
    ) -> str:
        document_path = self._conversation_path(tenant_id) / "documents" / f"{document_id}.txt"

        return document_path.read_text(
            encoding="utf-8",
        )

    def save_embeddings(
        self,
        tenant_id: UUID,
        document_id: UUID | str,
        embeddings: list[list[float]],
    ) -> Path:
        conversation_path = self.create_conversation_directory(tenant_id)
        embeddings_dir = conversation_path / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = embeddings_dir / f"{document_id}.json"
        embeddings_path.write_text(
            json.dumps(embeddings),
            encoding="utf-8",
        )
        return embeddings_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conversation_path(
        self,
        user_id: UUID,
    ) -> Path:
        if not user_id:
            raise ValueError("tenant_id cannot be empty")

        return self.base_path / str(user_id)
