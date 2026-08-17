from __future__ import annotations

from fastapi import APIRouter

from semantic_search_eng.retrival.retriver_manager import (
    RetriverManager,
)


router = APIRouter()

retriver_manager = RetriverManager()


# @router.post(
#     "/",
#     response_model=SearchResponse,
# )
# def search(
#     request: SearchRequest,
# ) -> SearchResponse:
#     try:
#         chunks = _load_all_chunks(
#             tenant_id=request.tenant_id,
#         )

#         embeddings = _load_all_embeddings(
#             tenant_id=request.tenant_id,
#             chunks=chunks,
#         )

#         results = retriver_manager.retrieve(
#             tenant_id=request.tenant_id,
#             query=request.query,
#             chunks=chunks,
#             embeddings=embeddings,
#             top_k=request.top_k,
#         )

#         return SearchResponse(
#             tenant_id=request.tenant_id,
#             query=request.query,
#             top_k=request.top_k,
#             results=results,
#         )

#     except FileNotFoundError as exc:
#         raise HTTPException(
#             status_code=404,
#             detail=str(exc),
#         ) from exc

#     except ValueError as exc:
#         raise HTTPException(
#             status_code=400,
#             detail=str(exc),
#         ) from exc

#     except Exception as exc:
#         raise HTTPException(
#             status_code=500,
#             detail=str(exc),
#         ) from exc


# def _load_all_chunks(
#     tenant_id: str,
# ) -> list[Chunk]:
#     conversation_path = data_manager.base_path / tenant_id / "chunks"

#     if not conversation_path.exists():
#         raise FileNotFoundError(f"Conversation not found: {tenant_id}")

#     chunks: list[Chunk] = []

#     for chunk_file in sorted(conversation_path.glob("*.json")):
#         stored_chunks = chunk_file.read_text(encoding="utf-8")

#         import json

#         raw_chunks = json.loads(stored_chunks)

#         chunks.extend(Chunk.model_validate(item) for item in raw_chunks)

#     return chunks


# def _load_all_embeddings(
#     tenant_id: str,
#     chunks: list[Chunk],
# ) -> list[list[float]]:
#     embeddings: list[list[float]] = []

#     document_ids: list[str] = []

#     for chunk in chunks:
#         if chunk.document_id not in document_ids:
#             document_ids.append(chunk.document_id)

#     for document_id in document_ids:
#         document_embeddings = data_manager.get_embeddings(
#             tenant_id=tenant_id,
#             document_id=document_id,
#         )

#         embeddings.extend(document_embeddings)

#     if len(chunks) != len(embeddings):
#         raise ValueError("Stored chunks and embeddings are out of sync.")

#     return embeddings
