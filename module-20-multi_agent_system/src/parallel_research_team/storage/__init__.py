"""Checkpoint persistence storage interfaces and adapters."""

from __future__ import annotations

from .file_store import FileCheckpointStore
from .protocol import CheckpointStore

__all__ = ["CheckpointStore", "FileCheckpointStore"]
