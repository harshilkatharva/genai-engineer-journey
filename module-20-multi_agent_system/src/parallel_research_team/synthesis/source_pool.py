"""Authoritative research evidence pool consolidating and deduplicating worker sources."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from ..models import SourceReference


class AuthoritativeSourcePool:
    """Thread-safe authoritative source repository for research synthesis and verification.

    Consolidates, validates, and deduplicates all SourceReference instances retrieved across
    completed and partial worker briefs, keyed by composite (source_id, chunk_id).
    """

    def __init__(self, sources: Iterable[SourceReference | dict[str, Any]] | None = None) -> None:
        self._lock = threading.RLock()
        self._sources: dict[tuple[str, str], SourceReference] = {}
        self._chunk_map: dict[str, SourceReference] = {}
        if sources:
            self.add_sources(sources)

    def _normalize_source(self, item: SourceReference | dict[str, Any]) -> SourceReference:
        """Convert a dict or SourceReference into a validated SourceReference instance."""
        if isinstance(item, SourceReference):
            return item
        if isinstance(item, dict):
            return SourceReference.model_validate(item)
        raise TypeError(f"Expected SourceReference or dict, got {type(item).__name__}")

    def add_source(self, source: SourceReference | dict[str, Any]) -> SourceReference:
        """Add or update a source chunk in the authoritative pool."""
        ref = self._normalize_source(source)
        with self._lock:
            key = (ref.source_id, ref.chunk_id)
            self._sources[key] = ref
            self._chunk_map[ref.chunk_id] = ref
        return ref

    def add_sources(
        self, sources: Iterable[SourceReference | dict[str, Any]]
    ) -> list[SourceReference]:
        """Bulk ingest and deduplicate sources into the pool."""
        added: list[SourceReference] = []
        for s in sources:
            added.append(self.add_source(s))
        return added

    def get(self, source_id: str, chunk_id: str) -> SourceReference | None:
        """Retrieve a specific chunk by (source_id, chunk_id) composite key."""
        with self._lock:
            return self._sources.get((source_id, chunk_id))

    def get_by_chunk_id(self, chunk_id: str) -> SourceReference | None:
        """Retrieve a source chunk by chunk_id alone."""
        with self._lock:
            return self._chunk_map.get(chunk_id)

    def has_source(self, source_id: str, chunk_id: str) -> bool:
        """Check if composite (source_id, chunk_id) exists in pool."""
        with self._lock:
            return (source_id, chunk_id) in self._sources

    def has_chunk(self, chunk_id: str) -> bool:
        """Check if chunk_id exists in pool."""
        with self._lock:
            return chunk_id in self._chunk_map

    def list_sources(self) -> list[SourceReference]:
        """Return a snapshot list of all pooled SourceReference instances."""
        with self._lock:
            return list(self._sources.values())

    def get_source_map(self) -> dict[tuple[str, str], SourceReference]:
        """Return a copy of the composite lookup dictionary."""
        with self._lock:
            return dict(self._sources)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize pooled sources into a dictionary keyed by chunk_id for checkpoint storage."""
        with self._lock:
            return {chunk_id: ref.model_dump() for chunk_id, ref in self._chunk_map.items()}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> AuthoritativeSourcePool:
        """Reconstruct an AuthoritativeSourcePool from serialized checkpoint data."""
        pool = cls()
        if data:
            pool.add_sources(data.values())
        return pool

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)
