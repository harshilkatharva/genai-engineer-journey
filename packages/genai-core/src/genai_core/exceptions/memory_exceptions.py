from __future__ import annotations


class MemoryError(Exception):
    """Base exception for all memory-related operations in genai-core."""


class MemoryStorageError(MemoryError):
    """Raised when an error occurs in a memory storage adapter or database operation."""


class MemoryNotFoundError(MemoryError):
    """Raised when a specified memory record, session, or resource is not found."""


class MemoryValidationError(MemoryError):
    """Raised when validation of memory content, parameters, or budget fails."""


class MemoryExtractionError(MemoryError):
    """Raised when extraction of knowledge from conversation fails or returns malformed data."""
