from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semantic_search_eng.config import get_settings
from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.retrival.retriver_manager import (
    RetriverManager,
)
from semantic_search_eng.user_data.data_manager import (
    DataManager,
)


class EvalutionManager:
    """
    Runs manually-labelled retrieval evaluation.

    Evaluation file format:

        [
            {
                "query": "...",
                "relevant_chunk_ids": [
                    "document_0000_chunk_0001"
                ]
            }
        ]
    """

    def __init__(
        self,
        evaluation_file: str | None = None,
    ) -> None:
        settings = get_settings()

        self.evaluation_file = Path(evaluation_file or settings.evaluation_file)

        self.data_manager = DataManager()
        self.retriver_manager = RetriverManager()

    def load_cases(
        self,
    ) -> list[dict[str, Any]]:
        if not self.evaluation_file.exists():
            raise FileNotFoundError(f"Evaluation file not found: {self.evaluation_file}")

        return json.loads(self.evaluation_file.read_text(encoding="utf-8"))

    def evaluate(
        self,
        conversation_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        cases = self.load_cases()

        chunks = self._load_chunks(conversation_id)

        embeddings = self._load_embeddings(
            conversation_id,
            chunks,
        )

        results: list[dict[str, Any]] = []

        hits = 0
        total_cases = len(cases)

        for case in cases:
            query = case["query"]

            relevant_ids = set(
                case.get(
                    "relevant_chunk_ids",
                    [],
                )
            )

            retrieved = self.retriver_manager.retrieve(
                conversation_id=conversation_id,
                query=query,
                chunks=chunks,
                embeddings=embeddings,
                top_k=top_k,
            )

            retrieved_ids = [result.chunk.chunk_id for result in retrieved]

            matched_ids = relevant_ids & set(retrieved_ids)

            hit = bool(matched_ids)

            if hit:
                hits += 1

            results.append(
                {
                    "query": query,
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "retrieved_chunk_ids": retrieved_ids,
                    "matched_chunk_ids": sorted(matched_ids),
                    "hit_at_k": hit,
                    "results": [
                        {
                            "chunk_id": result.chunk.chunk_id,
                            "similarity_score": (result.similarity_score),
                        }
                        for result in retrieved
                    ],
                }
            )

        recall_at_k = hits / total_cases if total_cases else 0.0

        return {
            "conversation_id": conversation_id,
            "top_k": top_k,
            "total_queries": total_cases,
            "successful_queries": hits,
            "recall_at_k": recall_at_k,
            "results": results,
        }

    def _load_chunks(
        self,
        conversation_id: str,
    ) -> list[Chunk]:
        chunks_directory = self.data_manager.base_path / conversation_id / "chunks"

        chunks: list[Chunk] = []

        for chunk_file in sorted(chunks_directory.glob("*.json")):
            raw_chunks = json.loads(chunk_file.read_text(encoding="utf-8"))

            chunks.extend(Chunk.model_validate(item) for item in raw_chunks)

        return chunks

    def _load_embeddings(
        self,
        conversation_id: str,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []

        document_ids: list[str] = []

        for chunk in chunks:
            if chunk.document_id not in document_ids:
                document_ids.append(chunk.document_id)

        for document_id in document_ids:
            embeddings.extend(
                self.data_manager.get_embeddings(
                    conversation_id=conversation_id,
                    document_id=document_id,
                )
            )

        if len(chunks) != len(embeddings):
            raise ValueError("Stored chunks and embeddings are out of sync.")

        return embeddings
