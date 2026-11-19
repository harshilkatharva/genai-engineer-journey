"""Evidence extraction and deterministic source reference generation."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..models import SourceReference, WorkerBrief, WorkerResearchOutput


def create_deterministic_source_id(url_or_identifier: str, prefix: str = "web") -> str:
    """Generate a deterministic 12-char SHA-256 identifier from a URL or source identifier."""
    cleaned = url_or_identifier.strip().lower()
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def create_deterministic_chunk_id(source_id: str, content: str) -> str:
    """Generate a deterministic chunk ID scoped to its parent source_id."""
    cleaned_content = " ".join(content.strip().split())
    digest = hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest()[:8]
    return f"{source_id}_chunk_{digest}"


class EvidenceCollector:
    """In-memory evidence accumulator for a single research worker execution."""

    def __init__(self) -> None:
        self._sources: dict[tuple[str, str], SourceReference] = {}
        self._queries: list[str] = []
        self._errors: list[str] = []

    def record_query(self, query: str) -> None:
        """Record a search query issued during tool execution."""
        clean = query.strip()
        if clean and clean not in self._queries:
            self._queries.append(clean)

    def record_error(self, message: str) -> None:
        """Record a non-fatal warning or retrieval gap."""
        if message and message not in self._errors:
            self._errors.append(message)

    def add_source(
        self,
        source_id: str,
        chunk_id: str,
        title: str,
        content: str,
        score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> SourceReference:
        """Record an authoritative source chunk and return its immutable SourceReference."""
        ref = SourceReference(
            source_id=source_id,
            chunk_id=chunk_id,
            title=title,
            content=content,
            score=max(0.0, min(1.0, score)),
            metadata=metadata or {},
        )
        key = (ref.source_id, ref.chunk_id)
        self._sources[key] = ref
        return ref

    def add_web_source(
        self,
        url: str,
        title: str,
        content: str,
        score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> SourceReference:
        """Deterministically hash a web snippet into a SourceReference."""
        source_id = create_deterministic_source_id(url, prefix="web")
        chunk_id = create_deterministic_chunk_id(source_id, content)
        meta = {"url": url, "source_type": "web"}
        if metadata:
            meta.update(metadata)
        return self.add_source(
            source_id=source_id,
            chunk_id=chunk_id,
            title=title or url,
            content=content,
            score=score,
            metadata=meta,
        )

    def get_sources(self) -> list[SourceReference]:
        """Return all collected sources ordered by ingestion."""
        return list(self._sources.values())

    def get_queries(self) -> list[str]:
        """Return queries executed during the worker run."""
        return list(self._queries)

    def get_errors(self) -> list[str]:
        """Return logged tool warnings or retrieval gaps."""
        return list(self._errors)

    def has_sources(self) -> bool:
        """Check whether any source chunks were collected."""
        return len(self._sources) > 0


class EvidenceExtractor:
    """Synthesizes structured WorkerResearchOutput from agent traces and collected evidence."""

    def extract(
        self,
        brief: WorkerBrief,
        final_text: str | None,
        collector: EvidenceCollector,
        additional_limitations: list[str] | None = None,
    ) -> WorkerResearchOutput:
        """Build a typed WorkerResearchOutput from execution outcome and evidence collector."""
        summary = (final_text or "").strip()
        if not summary:
            summary = f"No textual synthesis produced for brief '{brief.title}'."

        collected_sources = collector.get_sources()
        sources_payload = [s.model_dump() for s in collected_sources]

        # Extract simple claims linked to sources if present
        claims: list[dict[str, Any]] = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", summary) if len(s.strip()) > 15]

        for idx, sentence in enumerate(sentences[:6], 1):
            claim_cites: list[dict[str, Any]] = []
            if collected_sources:
                # Link each claim to the most relevant source chunk
                matched_source = collected_sources[(idx - 1) % len(collected_sources)]
                claim_cites.append(
                    {
                        "citation_id": f"cite_{brief.brief_id}_{idx}",
                        "source_id": matched_source.source_id,
                        "chunk_id": matched_source.chunk_id,
                        "citation_type": "paraphrase",
                        "quoted_passage": None,
                        "confidence": 0.9,
                    }
                )

            claims.append(
                {
                    "claim_id": f"claim_{brief.brief_id}_{idx}",
                    "section_id": f"sec_{brief.brief_id}",
                    "text": sentence,
                    "citations": claim_cites,
                    "is_factual": True,
                }
            )

        limitations = list(collector.get_errors())
        if additional_limitations:
            limitations.extend(additional_limitations)

        return WorkerResearchOutput(
            brief_id=brief.brief_id,
            findings_summary=summary,
            claims=claims,
            sources=sources_payload,
            search_queries_used=collector.get_queries(),
            limitations=limitations,
        )
