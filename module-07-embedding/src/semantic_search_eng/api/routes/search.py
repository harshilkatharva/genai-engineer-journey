from __future__ import annotations

from fastapi import APIRouter, HTTPException

from semantic_search_eng.chunking.chunking_manager import (
    ChunkingManager,
)
from semantic_search_eng.embedding.embedding_manager import (
    EmbeddingManager,
)
from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.models.process_request import (
    ProcessRequest,
)
from semantic_search_eng.models.search_request import (
    SearchRequest,
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

router = APIRouter()


data_manager = DataManager()
data_processor = DataProcessor()
chunking_manager = ChunkingManager()
embedding_manager = EmbeddingManager()
retriver_manager = RetriverManager()


@router.post(
    "/documents/process",
)
def process_documents(
    request: ProcessRequest,
) -> dict:
    """
    Process documents for a conversation:

        documents
            ↓
        store
            ↓
        chunk
            ↓
        embed
            ↓
        persist
    """
    try:
        document_ids = data_processor.process_documents(
            conversation_id=request.conversation_id,
            documents=request.documents,
        )

        document_texts: dict[str, str] = {}

        for document_id in document_ids:
            document_texts[document_id] = data_manager.get_document(
                conversation_id=request.conversation_id,
                document_id=document_id,
            )

        chunks = chunking_manager.chunk_documents(
            conversation_id=request.conversation_id,
            documents=document_texts,
        )

        # Store chunks grouped by document.
        grouped_chunks: dict[str, list[dict]] = {}

        for chunk in chunks:
            grouped_chunks.setdefault(
                chunk.document_id,
                [],
            ).append(chunk.model_dump())

        for document_id, document_chunks in grouped_chunks.items():
            data_manager.save_chunks(
                conversation_id=request.conversation_id,
                document_id=document_id,
                chunks=document_chunks,
            )

        embeddings = embedding_manager.embed_chunks(
            conversation_id=request.conversation_id,
            chunks=chunks,
            save=True,
        )

        return {
            "conversation_id": request.conversation_id,
            "document_count": len(document_ids),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "document_ids": document_ids,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.post(
    "/search",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
) -> SearchResponse:
    try:
        chunks = _load_all_chunks(
            conversation_id=request.conversation_id,
        )

        embeddings = _load_all_embeddings(
            conversation_id=request.conversation_id,
            chunks=chunks,
        )

        results = retriver_manager.retrieve(
            conversation_id=request.conversation_id,
            query=request.query,
            chunks=chunks,
            embeddings=embeddings,
            top_k=request.top_k,
        )

        return SearchResponse(
            conversation_id=request.conversation_id,
            query=request.query,
            top_k=request.top_k,
            results=results,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


def _load_all_chunks(
    conversation_id: str,
) -> list[Chunk]:
    conversation_path = data_manager.base_path / conversation_id / "chunks"

    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")

    chunks: list[Chunk] = []

    for chunk_file in sorted(conversation_path.glob("*.json")):
        stored_chunks = chunk_file.read_text(encoding="utf-8")

        import json

        raw_chunks = json.loads(stored_chunks)

        chunks.extend(Chunk.model_validate(item) for item in raw_chunks)

    return chunks


def _load_all_embeddings(
    conversation_id: str,
    chunks: list[Chunk],
) -> list[list[float]]:
    embeddings: list[list[float]] = []

    document_ids: list[str] = []

    for chunk in chunks:
        if chunk.document_id not in document_ids:
            document_ids.append(chunk.document_id)

    for document_id in document_ids:
        document_embeddings = data_manager.get_embeddings(
            conversation_id=conversation_id,
            document_id=document_id,
        )

        embeddings.extend(document_embeddings)

    if len(chunks) != len(embeddings):
        raise ValueError("Stored chunks and embeddings are out of sync.")

    return embeddings
