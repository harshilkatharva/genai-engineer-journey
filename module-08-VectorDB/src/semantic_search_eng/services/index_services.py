from semantic_search_eng.chunking.chunking_manager import (
    ChunkingManager,
)
from semantic_search_eng.embedding.embedding_manager import (
    EmbeddingManager,
)
from semantic_search_eng.models.process_request import (
    ProcessRequest,
)
from semantic_search_eng.models.search_response import (
    SearchResponse,
)
from semantic_search_eng.retrival.retriver_manager import (
    RetriverManager,
)
from semantic_search_eng.user_data.data_manager import (
    DataManager,
)
from semantic_search_eng.user_data.data_processor import (
    DataProcessor,
)

from semantic_search_eng.db.index_db import IndexDBManager


class IndexServiceManager:
    def __init__(self):
        self.data_manager = DataManager()
        self.data_processor = DataProcessor()
        self.chunking_manager = ChunkingManager()
        self.embedding_manager = EmbeddingManager()
        self.retriver_manager = RetriverManager()
        self.index_db_manager = IndexDBManager()

    async def index(self, request: ProcessRequest) -> SearchResponse:
        document_ids = self.data_processor.process_documents(
            tenant_id=request.tenant_id,
            documents=request.documents,
        )

        document_texts: dict[str, str] = {}
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

        embeddings = self.embedding_manager.embed_chunks(
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
