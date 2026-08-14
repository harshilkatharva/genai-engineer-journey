from __future__ import annotations

import re
from time import perf_counter

import tiktoken

from semantic_search_eng.config import get_settings
from semantic_search_eng.logger.chunking_tracker import (
    ChunkingTrackerLogger,
)
from semantic_search_eng.models.chunk import Chunk
from semantic_search_eng.models.chunking_tracker import ChunkingTracker

TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")


class ChunkingManager:
    """
    Creates sentence-boundary chunks using a configurable token target
    and token overlap.

    Default configuration:
        chunk_size   = 500 tokens
        chunk_overlap = 50 tokens
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tracker_logger = ChunkingTrackerLogger()
        self.tiktoken = tiktoken.get_encoding("cl100k_base")

    def chunk_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        started_at = perf_counter()

        sentences = self._split_sentences(text)
        chunks = self._build_chunks(
            tenant_id=tenant_id,
            document_id=document_id,
            sentences=sentences,
        )

        latency_ms = (perf_counter() - started_at) * 1000
        total_tokens = sum(chunk.token_count for chunk in chunks)

        tracker = ChunkingTracker(
            tenant_id=tenant_id,
            document_count=1,
            total_chunks=len(chunks),
            chunking_strategy=self.settings.chunking_strategy,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            total_input_tokens=total_tokens,
            latency_ms=latency_ms,
        )

        self.tracker_logger.track(tracker)

        return chunks

    def chunk_documents(
        self,
        tenant_id: str,
        documents: dict[str, str],
        document_type: str,
        metadata: dict,
    ) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            tenant_id: Conversation identifier.
            documents: Mapping of document_id -> document text.
        """
        started_at = perf_counter()

        all_chunks: list[Chunk] = []

        for document_id, text in documents.items():
            all_chunks.extend(
                self._build_chunks(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_type=document_type,
                    metadata=metadata,
                    sentences=self._split_sentences(text),
                )
            )

        latency_ms = (perf_counter() - started_at) * 1000
        total_tokens = sum(chunk.token_count for chunk in all_chunks)

        tracker = ChunkingTracker(
            tenant_id=tenant_id,
            document_count=len(documents),
            total_chunks=len(all_chunks),
            chunking_strategy=self.settings.chunking_strategy,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            total_input_tokens=total_tokens,
            latency_ms=latency_ms,
        )

        self.tracker_logger.track(tracker)

        return all_chunks

    def _build_chunks(
        self,
        tenant_id: str,
        document_id: str,
        document_type: str,
        metadata: dict,
        sentences: list[tuple[str, int, int]],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        current_sentences: list[tuple[str, int, int, int]] = []
        current_tokens = 0
        chunk_index = 0

        for sentence, start, end in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # A single oversized sentence cannot be split while preserving
            # sentence boundaries. It therefore becomes its own chunk.
            if current_sentences and current_tokens + sentence_tokens > self.settings.chunk_size:
                chunks.append(
                    self._create_chunk(
                        tenant_id=tenant_id,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        document_type=document_type,
                        metadata=metadata,
                        sentences=current_sentences,
                    )
                )

                chunk_index += 1

                overlap_sentences = self._get_overlap_sentences(current_sentences)

                current_sentences = overlap_sentences
                current_tokens = sum(item[3] for item in current_sentences)

            current_sentences.append(
                (
                    sentence,
                    start,
                    end,
                    sentence_tokens,
                )
            )
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(
                self._create_chunk(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    document_type=document_type,
                    metadata=metadata,
                    sentences=current_sentences,
                )
            )

        return chunks

    def _create_chunk(
        self,
        tenant_id: str,
        document_id: str,
        chunk_index: int,
        document_type: str,
        metadata: dict,
        sentences: list[tuple[str, int, int, int]],
    ) -> Chunk:
        text = " ".join(sentence[0].strip() for sentence in sentences).strip()

        start_position = sentences[0][1]
        end_position = sentences[-1][2]

        return Chunk(
            chunk_id=(f"{document_id}_chunk_{chunk_index:04d}"),
            document_id=document_id,
            tenant_id=tenant_id,
            document_type=document_type,
            metadata=metadata,
            chunk_index=chunk_index,
            text=text,
            token_count=self._estimate_tokens(text),
            start_position=start_position,
            end_position=end_position,
        )

    def _get_overlap_sentences(
        self,
        sentences: list[tuple[str, int, int, int]],
    ) -> list[tuple[str, int, int, int]]:
        """
        Keep trailing complete sentences until the configured overlap
        is approximately reached.

        We intentionally preserve sentence boundaries rather than cutting
        a sentence in the middle.
        """
        overlap_tokens = self.settings.chunk_overlap

        if overlap_tokens <= 0:
            return []

        selected: list[tuple[str, int, int, int]] = []
        selected_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = sentence[3]

            selected.insert(0, sentence)
            selected_tokens += sentence_tokens

            if selected_tokens >= overlap_tokens:
                break

        return selected

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[tuple[str, int, int]]:
        """
        Lightweight sentence segmentation.

        Returns:
            (sentence_text, start_position, end_position)
        """
        normalized = text.strip()

        if not normalized:
            return []

        pattern = re.compile(
            r".+?(?:"
            r"(?<=[.!?])(?=\s+)"
            r"|(?=\n{2,})"
            r"|$"
            r")",
            re.DOTALL,
        )

        sentences: list[tuple[str, int, int]] = []

        for match in pattern.finditer(normalized):
            sentence = match.group(0).strip()

            if not sentence:
                continue

            start = match.start()
            end = match.end()

            while start < end and normalized[start].isspace():
                start += 1

            while end > start and normalized[end - 1].isspace():
                end -= 1

            sentences.append(
                (
                    normalized[start:end],
                    start,
                    end,
                )
            )

        return sentences

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Count tokens using tiktoken.

        We use cl100k_base as the tokenizer for chunk-size accounting.
        This is used only for chunking/token tracking; the actual
        embedding model remains all-MiniLM-L6-v2.
        """
        if not text.strip():
            return 0

        return len(
            TOKEN_ENCODER.encode(
                text,
                disallowed_special=(),
            )
        )
