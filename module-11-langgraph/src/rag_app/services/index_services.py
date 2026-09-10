from uuid import UUID

from rag_app.chunking.chunking_manager import (
    ChunkingManager,
)
from rag_app.db.index_db import IndexDBManager
from rag_app.embedding.embedding_manager import (
    EmbeddingManager,
)
from rag_app.models import ProcessRequest
from rag_app.user_data.data_manager import DataManager
from rag_app.user_data.data_processor import (
    DataProcessor,
)


class IndexServiceManager:
    def __init__(self):
        self.data_manager = DataManager()
        self.data_processor = DataProcessor()
        self.chunking_manager = ChunkingManager()
        self.embedding_manager = EmbeddingManager()
        self.index_db_manager = IndexDBManager()

    async def index(self, request: ProcessRequest):
        document_ids = self.data_processor.process_documents(
            tenant_id=request.tenant_id,
            documents=request.documents,
        )

        document_texts: dict[UUID, str] = {}
        for document_id in document_ids:
            document_texts[document_id] = self.data_manager.get_document(
                tenant_id=request.tenant_id,
                document_id=document_id,
            )

        chunks = self.chunking_manager.chunk_documents(
            tenant_id=request.tenant_id,
            documents=document_texts,
            document_type=request.documents_type,
            metadata=request.meta_data,
        )

        embeddings = await self.embedding_manager.embed_chunks(
            tenant_id=request.tenant_id,
            chunks=chunks,
        )

        await self.index_db_manager.store_index(
            tenant_id=request.tenant_id, chunks=chunks, embeddings=embeddings
        )

        return {
            "tenant_id": request.tenant_id,
            "document_count": len(document_ids),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "document_ids": document_ids,
        }
