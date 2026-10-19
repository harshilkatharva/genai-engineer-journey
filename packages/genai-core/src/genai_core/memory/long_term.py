from __future__ import annotations

import contextlib
import logging
import math
from typing import TYPE_CHECKING, Any

from ..core import get_settings
from ..exceptions import MemoryExtractionError, MemoryValidationError
from ..models.memory_model import (
    ConsolidatedMemoryItem,
    ConsolidationResult,
    ConsolidationStatus,
    ConversationMessage,
    LongTermMemoryRecord,
    MemoryCandidate,
    MemoryCategory,
    MemorySearchResult,
)
from ..observability import ObservabilityEvent, get_logger, safe_log
from .extraction import LLMMemoryExtractor, validate_memory_content
from .protocol import MemoryStorageAdapter

if TYPE_CHECKING:
    from ..services.embedding_service import EmbeddingService
    from ..services.llm_services import LLMService

_logger = get_logger("genai_core.memory.long_term")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class LongTermMemory:
    """Manages persistent facts, preferences, profiles, and semantic memory search with recency decay."""

    def __init__(
        self,
        store: MemoryStorageAdapter,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
        recency_half_life_days: int | None = None,
    ) -> None:
        self._store = store
        self.store_adapter = store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        settings = get_settings()
        self.recency_half_life_days = (
            recency_half_life_days
            if recency_half_life_days is not None
            else settings.memory_recency_half_life_days
        )

    async def store(
        self,
        user_id: str,
        candidate: MemoryCandidate | str,
        **kwargs: Any,
    ) -> LongTermMemoryRecord:
        """Store or update a persistent memory record with semantic vector embedding."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")

        if isinstance(candidate, str):
            cand = MemoryCandidate(content=candidate, **kwargs)
        elif isinstance(candidate, MemoryCandidate):
            cand = candidate
        else:
            raise TypeError(
                f"Expected candidate to be str or MemoryCandidate, got {type(candidate).__name__}"
            )

        content = cand.content.strip()
        if not (1 <= len(content) <= 4000):
            raise MemoryValidationError(
                f"Memory content length ({len(content)}) must be between 1 and 4000 characters"
            )

        # Generate embedding vector if embedding service is configured
        vector: list[float] | None = None
        if self.embedding_service is not None:
            vector = await self.embedding_service.embed_query(content)

        record = LongTermMemoryRecord(
            user_id=user_id.strip(),
            content=content,
            category=cand.category,
            confidence=cand.confidence,
            supersedes_ids=cand.supersedes_ids,
            expires_at=cand.expires_at,
            metadata=cand.metadata,
            embedding=vector,
        )

        saved = await self._store.save_record(record)
        safe_log(
            _logger,
            logging.INFO,
            "Long-term memory stored successfully",
            event=ObservabilityEvent.MEMORY_STORE_SUCCESS,
            user_id=user_id,
            memory_id=saved.id,
            category=str(saved.category),
        )
        return saved

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        category: MemoryCategory | str | None = None,
        similarity_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[MemorySearchResult]:
        """Search user memories using cosine similarity and exponential recency decay."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")
        if not query or not query.strip():
            raise MemoryValidationError("Search query cannot be empty")
        if top_k <= 0:
            raise MemoryValidationError("top_k must be a positive integer")

        if self.embedding_service is None:
            raise MemoryValidationError("EmbeddingService is required for semantic memory search")

        query_vector = await self.embedding_service.embed_query(query.strip())
        results = await self._store.search_records(
            user_id=user_id.strip(),
            query_embedding=query_vector,
            top_k=top_k,
            half_life_days=self.recency_half_life_days,
            similarity_threshold=similarity_threshold,
            category=category,
            include_superseded=include_superseded,
        )

        safe_log(
            _logger,
            logging.INFO,
            "Long-term memory search completed",
            event=ObservabilityEvent.MEMORY_RETRIEVE_SUCCESS,
            user_id=user_id,
            results_count=len(results),
            top_k=top_k,
        )
        return results

    async def view(
        self,
        user_id: str,
        category: MemoryCategory | str | None = None,
        include_superseded: bool = False,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[LongTermMemoryRecord]:
        """List active long-term memories for a user."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")
        return await self._store.list_records(
            user_id=user_id.strip(),
            category=category,
            include_superseded=include_superseded,
            include_expired=include_expired,
            limit=limit,
        )

    async def delete(self, user_id: str, record_id: str) -> bool:
        """Delete a specific long-term memory record."""
        if not user_id or not record_id:
            raise MemoryValidationError("user_id and record_id are required")
        deleted = await self._store.delete_record(
            user_id=user_id.strip(), record_id=record_id.strip()
        )
        if deleted:
            safe_log(
                _logger,
                logging.INFO,
                "Long-term memory record deleted",
                event=ObservabilityEvent.MEMORY_DELETE_SUCCESS,
                user_id=user_id,
                record_id=record_id,
            )
        return deleted

    async def purge_expired(self, user_id: str | None = None) -> int:
        """Purge expired memory records."""
        count = await self._store.purge_expired(user_id=user_id.strip() if user_id else None)
        safe_log(
            _logger,
            logging.INFO,
            "Expired memory records purged",
            event=ObservabilityEvent.MEMORY_PURGE_SUCCESS,
            purged_count=count,
            user_id=user_id,
        )
        return count

    async def consolidate(
        self,
        user_id: str,
        messages: list[ConversationMessage],
        existing_memories: list[LongTermMemoryRecord] | None = None,
        custom_prompt_template: str | None = None,
        llm_service: LLMService | None = None,
        llm_provider: str | None = None,
        similarity_threshold: float = 0.95,
        session_id: str | None = None,
    ) -> ConsolidationResult:
        """Extract durable knowledge from conversation turns and persist valid memories with deduplication."""
        if not user_id or not user_id.strip():
            raise MemoryValidationError("user_id is required")

        uid = user_id.strip()
        safe_log(
            _logger,
            logging.INFO,
            "Starting automatic knowledge consolidation",
            event=ObservabilityEvent.MEMORY_CONSOLIDATION_START,
            user_id=uid,
            session_id=session_id,
            messages_count=len(messages),
        )

        if not messages:
            return ConsolidationResult(
                user_id=uid,
                session_id=session_id,
                total_extracted=0,
                total_processed=0,
            )

        effective_llm_service = llm_service or self.llm_service
        if effective_llm_service is None:
            raise MemoryValidationError(
                "LLMService is required for automatic knowledge consolidation"
            )

        # Retrieve active existing memories for user if not provided
        if existing_memories is None:
            active_existing = await self.view(
                user_id=uid,
                include_superseded=False,
                include_expired=False,
                limit=500,
            )
        else:
            active_existing = list(existing_memories)

        extractor = LLMMemoryExtractor(effective_llm_service)
        try:
            candidates = await extractor.extract(
                messages=messages,
                existing_memories=active_existing,
                custom_prompt_template=custom_prompt_template,
                llm_provider=llm_provider,
            )
        except Exception as err:
            safe_log(
                _logger,
                logging.ERROR,
                f"Memory extraction failed: {err}",
                event=ObservabilityEvent.MEMORY_CONSOLIDATION_FAILED,
                user_id=uid,
                error=str(err),
            )
            raise

        created: list[LongTermMemoryRecord] = []
        updated: list[LongTermMemoryRecord] = []
        skipped: list[ConsolidatedMemoryItem] = []
        failed: list[ConsolidatedMemoryItem] = []

        existing_ids = {str(rec.id) for rec in active_existing}

        for cand in candidates:
            # 1. Validate sensitive information
            try:
                validate_memory_content(cand.content)
            except MemoryExtractionError as err:
                skipped.append(
                    ConsolidatedMemoryItem(
                        status=ConsolidationStatus.SKIPPED,
                        candidate=cand,
                        reason=f"Sensitive pattern detected: {err}",
                    )
                )
                continue

            # 2. Check for exact duplicate in active memories
            exact_dup = next(
                (
                    rec
                    for rec in active_existing
                    if rec.content.strip().casefold() == cand.content.strip().casefold()
                    and str(rec.category) == str(cand.category)
                ),
                None,
            )
            if exact_dup is not None and not cand.supersedes_ids:
                skipped.append(
                    ConsolidatedMemoryItem(
                        status=ConsolidationStatus.SKIPPED,
                        candidate=cand,
                        reason=f"Exact duplicate of existing record {exact_dup.id}",
                    )
                )
                continue

            # 3. Check for high semantic similarity duplicate (if embedding service is active and not superseding)
            cand_vector: list[float] | None = None
            if self.embedding_service is not None and not cand.supersedes_ids:
                with contextlib.suppress(Exception):
                    cand_vector = await self.embedding_service.embed_query(cand.content.strip())
                    sem_dup = None
                    max_sim = 0.0
                    for rec in active_existing:
                        if rec.embedding is not None and str(rec.category) == str(cand.category):
                            sim = _cosine_similarity(cand_vector, rec.embedding)
                            if sim > max_sim:
                                max_sim = sim
                                if sim >= similarity_threshold:
                                    sem_dup = rec
                                    break
                    if sem_dup is not None:
                        skipped.append(
                            ConsolidatedMemoryItem(
                                status=ConsolidationStatus.SKIPPED,
                                candidate=cand,
                                reason=f"High semantic similarity ({max_sim:.2f}) with existing record {sem_dup.id}",
                            )
                        )
                        continue

            # 4. Filter and sanitize supersedes_ids to ensure they reference existing memories
            valid_supersedes: list[str] = []
            if cand.supersedes_ids:
                for sid in cand.supersedes_ids:
                    if sid in existing_ids:
                        valid_supersedes.append(sid)
                cand.supersedes_ids = valid_supersedes

            # 5. Store valid candidate
            try:
                saved_record = await self.store(user_id=uid, candidate=cand)
                if cand.supersedes_ids:
                    updated.append(saved_record)
                else:
                    created.append(saved_record)

                # Add saved record to active_existing for intra-batch deduplication
                active_existing.append(saved_record)
                existing_ids.add(str(saved_record.id))
            except Exception as err:  # noqa: BLE001
                failed.append(
                    ConsolidatedMemoryItem(
                        status=ConsolidationStatus.FAILED,
                        candidate=cand,
                        reason=f"Storage failure: {err}",
                    )
                )

        result = ConsolidationResult(
            user_id=uid,
            session_id=session_id,
            created=created,
            updated=updated,
            skipped=skipped,
            failed=failed,
            total_extracted=len(candidates),
            total_processed=len(created) + len(updated) + len(skipped) + len(failed),
            metadata={
                "provider": llm_provider or "default",
                "extracted_count": len(candidates),
                "created_count": len(created),
                "updated_count": len(updated),
                "skipped_count": len(skipped),
                "failed_count": len(failed),
            },
        )

        safe_log(
            _logger,
            logging.INFO,
            "Knowledge consolidation completed successfully",
            event=ObservabilityEvent.MEMORY_CONSOLIDATION_SUCCESS,
            user_id=uid,
            created_count=len(created),
            updated_count=len(updated),
            skipped_count=len(skipped),
            failed_count=len(failed),
        )

        return result
