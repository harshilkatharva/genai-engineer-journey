"""Thread-safe extensible worker registry."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from genai_core.observability import get_logger

from ..core.config import ResearchTeamConfig
from ..exceptions import WorkerExecutionError
from ..models import WorkerBrief
from .protocol import ResearchWorkerProtocol

_logger = get_logger("parallel_research_team.workers.registry")

WorkerFactory = Callable[..., ResearchWorkerProtocol]


class WorkerRegistry:
    """Thread-safe factory registry for creating research workers.

    Allows plugging in new specialized worker types (e.g. financial analyst,
    code researcher, policy auditor) without modifying scheduler or orchestrator logic.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._factories: dict[str, WorkerFactory] = {}

    def register(
        self,
        worker_type: str,
        factory: WorkerFactory,
        override: bool = False,
    ) -> None:
        """Register a worker factory for a given worker_type."""
        cleaned_type = worker_type.strip().lower()
        if not cleaned_type:
            raise ValueError("worker_type cannot be empty.")
        if not callable(factory):
            raise TypeError(f"factory for '{cleaned_type}' must be callable.")

        with self._lock:
            if cleaned_type in self._factories and not override:
                raise ValueError(
                    f"Worker type '{cleaned_type}' is already registered. "
                    "Set override=True to replace it."
                )
            self._factories[cleaned_type] = factory
            _logger.info("Registered worker factory for type: '%s'", cleaned_type)

    def unregister(self, worker_type: str) -> WorkerFactory | None:
        """Remove a worker factory from the registry."""
        cleaned_type = worker_type.strip().lower()
        with self._lock:
            return self._factories.pop(cleaned_type, None)

    def get(self, worker_type: str) -> WorkerFactory | None:
        """Retrieve a registered worker factory by type."""
        cleaned_type = worker_type.strip().lower()
        with self._lock:
            return self._factories.get(cleaned_type)

    def has(self, worker_type: str) -> bool:
        """Check if a worker type is registered."""
        cleaned_type = worker_type.strip().lower()
        with self._lock:
            return cleaned_type in self._factories

    def list_types(self) -> list[str]:
        """Return a snapshot list of all registered worker types."""
        with self._lock:
            return list(self._factories.keys())

    def create_worker(
        self,
        brief: WorkerBrief,
        config: ResearchTeamConfig,
        **kwargs: Any,
    ) -> ResearchWorkerProtocol:
        """Instantiate a worker for the given brief using registered factories."""
        requested_type = brief.worker_type.strip().lower()

        with self._lock:
            factory = self._factories.get(requested_type)
            if factory is None and requested_type != "default":
                _logger.warning(
                    "Worker type '%s' requested by brief '%s' not found. Falling back to 'default'.",
                    requested_type,
                    brief.brief_id,
                )
                factory = self._factories.get("default")

        if factory is None:
            raise WorkerExecutionError(
                f"No worker factory registered for type '{brief.worker_type}' "
                "and no 'default' factory is available."
            )

        return factory(brief=brief, config=config, **kwargs)


# Global default registry instance
_default_registry: WorkerRegistry | None = None
_registry_lock = threading.Lock()


def get_default_worker_registry() -> WorkerRegistry:
    """Return the process-wide default WorkerRegistry singleton."""
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = WorkerRegistry()
        return _default_registry
