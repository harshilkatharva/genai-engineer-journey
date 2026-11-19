"""Unit tests for ResearchWorkerProtocol and WorkerRegistry."""

from __future__ import annotations

import pytest

from parallel_research_team.core.budget import SharedRunBudget
from parallel_research_team.core.config import ResearchTeamConfig
from parallel_research_team.exceptions import WorkerExecutionError
from parallel_research_team.models import (
    WorkerBrief,
    WorkerExecutionResult,
    WorkerStatus,
)
from parallel_research_team.workers.protocol import ResearchWorkerProtocol
from parallel_research_team.workers.registry import (
    WorkerRegistry,
    get_default_worker_registry,
)


class DummyWorker:
    """Dummy worker for testing protocol compliance."""

    @property
    def worker_type(self) -> str:
        return "dummy"

    async def execute(
        self,
        brief: WorkerBrief,
        shared_budget: SharedRunBudget,
    ) -> WorkerExecutionResult:
        return WorkerExecutionResult(
            brief_id=brief.brief_id,
            status=WorkerStatus.COMPLETED,
        )


def test_protocol_runtime_check() -> None:
    worker = DummyWorker()
    assert isinstance(worker, ResearchWorkerProtocol)


def test_registry_registration_and_lookup() -> None:
    registry = WorkerRegistry()
    registry.register("dummy", lambda brief, config, **kwargs: DummyWorker())

    assert registry.has("dummy")
    assert "dummy" in registry.list_types()

    brief = WorkerBrief(
        brief_id="b1",
        title="Test Brief",
        topic="Testing",
        objective="Verify registry",
        task_boundaries=["Focus on tests", "Do not ignore errors"],
        worker_type="dummy",
    )
    worker = registry.create_worker(brief=brief, config=ResearchTeamConfig())
    assert worker.worker_type == "dummy"


def test_registry_fallback_to_default() -> None:
    registry = WorkerRegistry()
    registry.register("default", lambda brief, config, **kwargs: DummyWorker())

    # Request unregistered worker_type
    brief = WorkerBrief(
        brief_id="b2",
        title="Custom Brief",
        topic="Testing",
        objective="Verify fallback",
        task_boundaries=["Focus on tests", "Do not ignore errors"],
        worker_type="unknown_specialist",
    )
    worker = registry.create_worker(brief=brief, config=ResearchTeamConfig())
    assert worker.worker_type == "dummy"


def test_registry_missing_worker_type_and_no_default_raises() -> None:
    registry = WorkerRegistry()
    brief = WorkerBrief(
        brief_id="b3",
        title="Brief",
        topic="Testing",
        objective="Verify error",
        task_boundaries=["Focus on tests", "Do not ignore errors"],
        worker_type="non_existent",
    )
    with pytest.raises(WorkerExecutionError, match="No worker factory registered"):
        registry.create_worker(brief=brief, config=ResearchTeamConfig())


def test_registry_duplicate_registration_without_override_raises() -> None:
    registry = WorkerRegistry()
    registry.register("dummy", lambda brief, config, **kwargs: DummyWorker())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", lambda brief, config, **kwargs: DummyWorker())


def test_registry_unregister() -> None:
    registry = WorkerRegistry()
    registry.register("temp", lambda brief, config, **kwargs: DummyWorker())
    assert registry.has("temp")
    removed = registry.unregister("temp")
    assert removed is not None
    assert not registry.has("temp")


def test_default_worker_registry_has_default() -> None:
    reg = get_default_worker_registry()
    assert reg.has("default")
