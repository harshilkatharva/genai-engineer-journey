"""Workers package for bounded parallel research agents."""

from .extractor import (
    EvidenceCollector,
    EvidenceExtractor,
    create_deterministic_chunk_id,
    create_deterministic_source_id,
)
from .protocol import ResearchWorkerProtocol
from .registry import (
    WorkerFactory,
    WorkerRegistry,
    get_default_worker_registry,
)
from .tools import (
    create_research_tool_registry,
    create_retrieval_service,
)
from .worker import (
    DefaultResearchWorker,
    default_worker_factory,
)

# Register default worker factory and standard brief aliases into the default registry
for alias in [
    "default",
    "factual",
    "comparative",
    "exploratory",
    "deep_dive",
    "factual_retrieval",
    "comparative_analysis",
    "exploratory_deep_dive",
]:
    get_default_worker_registry().register(
        alias,
        default_worker_factory,
        override=True,
    )

__all__ = [
    "DefaultResearchWorker",
    "EvidenceCollector",
    "EvidenceExtractor",
    "ResearchWorkerProtocol",
    "WorkerFactory",
    "WorkerRegistry",
    "create_deterministic_chunk_id",
    "create_deterministic_source_id",
    "create_research_tool_registry",
    "create_retrieval_service",
    "default_worker_factory",
    "get_default_worker_registry",
]
