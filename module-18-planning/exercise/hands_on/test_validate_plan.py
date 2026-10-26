from enum import Enum

import pytest
from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    id: str = Field(description="unique id for each plansstep of plan")
    description: str = Field(description="Description for planstep")
    tool_name_hint: str | None = Field(default=None, description="Tool hint for this step")
    depends_on: list[str] | None = Field(
        default_factory=list, description="Id's of previous depended steps"
    )
    success_criteria: str = Field(description="criterias for mark success")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Status of this step")
    result_summary: str | None = Field(default=None, description="Summary of this step")
    attempts: int = Field(default=0, description="Attempts to complete this step")


class Plan(BaseModel):
    goal: str = Field(description="Goal for the agent")
    steps: list[PlanStep] = Field(description="Steps to complete goal")
    version: int = Field(default=1, description="Version of this plan")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions for this goals")


class PlanValidationError(Exception):
    """Base exception for all plan validation errors."""


class EmptyPlanError(PlanValidationError):
    """Raised when a plan contains no steps."""


class DuplicateStepIdError(PlanValidationError):
    """Raised when multiple steps share the same ID."""


class MissingDependencyError(PlanValidationError):
    """Raised when a step depends on an ID that does not exist in the plan."""


class SelfDependencyError(PlanValidationError):
    """Raised when a step lists itself as a dependency."""


class CircularDependencyError(PlanValidationError):
    """Raised when a dependency cycle is detected (plan is not a DAG)."""


def validate_plan(plan: Plan) -> bool:
    """
    Validates that a Plan satisfies all structural integrity rules:
      1. Plan has at least one step.
      2. Step IDs are unique.
      3. No step depends on itself.
      4. All dependencies exist in the plan.
      5. The dependency graph has no cycles (is a valid DAG).

    Returns:
        bool: True if the plan is valid.

    Raises:
        EmptyPlanError: If plan has no steps.
        DuplicateStepIdError: If duplicate step IDs exist.
        SelfDependencyError: If a step depends on itself.
        MissingDependencyError: If a referenced dependency ID is missing.
        CircularDependencyError: If a dependency cycle is detected.
    """
    # 1. Check for empty plan
    if not plan.steps:
        raise EmptyPlanError("Plan must contain at least one step.")

    # 2. Check for duplicate IDs
    seen_ids: set[str] = set()
    step_map: dict[str, PlanStep] = {}
    for step in plan.steps:
        if step.id in seen_ids:
            raise DuplicateStepIdError(f"Duplicate step ID found: '{step.id}'")
        seen_ids.add(step.id)
        step_map[step.id] = step

    all_step_ids = seen_ids

    # 3. Check for self-dependencies and missing dependencies
    for step in plan.steps:
        deps = step.depends_on or []
        for dep_id in deps:
            if dep_id == step.id:
                raise SelfDependencyError(f"Step '{step.id}' cannot depend on itself.")
            if dep_id not in all_step_ids:
                raise MissingDependencyError(
                    f"Step '{step.id}' depends on non-existent step '{dep_id}'."
                )

    # 4. Cycle detection using Depth-First Search (DFS with 3-state tracking)
    # States: 0 = UNVISITED, 1 = VISITING (in recursion stack), 2 = VISITED (completed)
    UNVISITED = 0
    VISITING = 1
    VISITED = 2

    state: dict[str, int] = {s_id: UNVISITED for s_id in all_step_ids}

    def dfs(node: str, path: list[str]) -> None:
        state[node] = VISITING
        path.append(node)

        deps = step_map[node].depends_on or []
        for dep in deps:
            if state[dep] == VISITING:
                cycle_start = path.index(dep)
                cycle = path[cycle_start:] + [dep]
                raise CircularDependencyError(f"Circular dependency detected: {' -> '.join(cycle)}")
            elif state[dep] == UNVISITED:
                dfs(dep, path)

        path.pop()
        state[node] = VISITED

    for step_id in all_step_ids:
        if state[step_id] == UNVISITED:
            dfs(step_id, [])

    return True


def test_valid_plan():
    valid_plan = Plan(
        goal="Test Goal",
        steps=[
            PlanStep(id="step_1", description="Step 1", success_criteria="Done"),
            PlanStep(
                id="step_2",
                description="Step 2",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_3",
                description="Step 3",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_4",
                description="Step 4",
                depends_on=["step_2", "step_3"],
                success_criteria="Done",
            ),
        ],
    )
    assert validate_plan(valid_plan) is True


def test_empty_plan_error():
    empty_plan = Plan(goal="Empty", steps=[])
    with pytest.raises(EmptyPlanError):
        validate_plan(empty_plan)


def test_duplicate_step_id_error():
    plan = Plan(
        goal="Duplicate IDs",
        steps=[
            PlanStep(id="step_1", description="First", success_criteria="Done"),
            PlanStep(id="step_1", description="Second", success_criteria="Done"),
        ],
    )
    with pytest.raises(DuplicateStepIdError):
        validate_plan(plan)


def test_missing_dependency_error():
    plan = Plan(
        goal="Missing Dep",
        steps=[
            PlanStep(
                id="step_1",
                description="First",
                depends_on=["step_ghost"],
                success_criteria="Done",
            )
        ],
    )
    with pytest.raises(MissingDependencyError):
        validate_plan(plan)


def test_self_dependency_error():
    plan = Plan(
        goal="Self Dep",
        steps=[
            PlanStep(
                id="step_1",
                description="First",
                depends_on=["step_1"],
                success_criteria="Done",
            )
        ],
    )
    with pytest.raises(SelfDependencyError):
        validate_plan(plan)


def test_circular_dependency_direct():
    plan = Plan(
        goal="Cycle 2 nodes",
        steps=[
            PlanStep(
                id="step_1",
                description="Step 1",
                depends_on=["step_2"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_2",
                description="Step 2",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
        ],
    )
    with pytest.raises(CircularDependencyError) as exc_info:
        validate_plan(plan)
    assert "Circular dependency detected" in str(exc_info.value)


def test_circular_dependency_indirect():
    plan = Plan(
        goal="Cycle 3 nodes",
        steps=[
            PlanStep(
                id="step_1",
                description="Step 1",
                depends_on=["step_3"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_2",
                description="Step 2",
                depends_on=["step_1"],
                success_criteria="Done",
            ),
            PlanStep(
                id="step_3",
                description="Step 3",
                depends_on=["step_2"],
                success_criteria="Done",
            ),
        ],
    )
    with pytest.raises(CircularDependencyError) as exc_info:
        validate_plan(plan)
    assert "Circular dependency detected" in str(exc_info.value)


def test_llm_style_realistic_plan():
    realistic_plan = Plan(
        goal="Build and deploy a containerized FastAPI microservice on AWS",
        steps=[
            PlanStep(
                id="step_1",
                description="Develop FastAPI application",
                tool_name_hint="IDE",
                success_criteria="Passing unit tests",
            ),
            PlanStep(
                id="step_2",
                description="Create Dockerfile",
                tool_name_hint="Docker",
                depends_on=["step_1"],
                success_criteria="Builds successfully",
            ),
            PlanStep(
                id="step_3",
                description="Provision AWS ECS and RDS with Terraform",
                tool_name_hint="Terraform",
                success_criteria="Resources created",
            ),
            PlanStep(
                id="step_4",
                description="Setup GitHub Actions CI/CD",
                tool_name_hint="GitHub Actions",
                depends_on=["step_2", "step_3"],
                success_criteria="Workflow green",
            ),
        ],
    )
    assert validate_plan(realistic_plan) is True
