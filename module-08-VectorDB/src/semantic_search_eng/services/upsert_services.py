from semantic_search_eng.db.upsert_db import UpsertDBManager
from semantic_search_eng.embedding.embedding_manager import (
    EmbeddingManager,
)
from semantic_search_eng.models import UpsertRequest
from semantic_search_eng.user_data.data_manager import DataManager
from semantic_search_eng.user_data.data_processor import (
    DataProcessor,
)


class UpsertServiceManager:
    def __init__(self):
        self.data_manager = DataManager()
        self.data_processor = DataProcessor()
        self.embedding_manager = EmbeddingManager()
        self.upsert_db_manager = UpsertDBManager()

    async def upsert_chunks(self, request: UpsertRequest):
        embedding = await self.embedding_manager.embed_chunks(
            tenant_id=request.tenant_id,
            chunks=request.updated_chunks,
        )

        updated_chunk_ids = await self.upsert_db_manager.upsert_chunks(
            tenant_id=request.tenant_id,
            updated_chunks=request.updated_chunks,
            updated_embeddings=embedding,
            document_ids=request.document_ids,
            chunk_ids=request.chunk_ids,
        )

        return updated_chunk_ids
