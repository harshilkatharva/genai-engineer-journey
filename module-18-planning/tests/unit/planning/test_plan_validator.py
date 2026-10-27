from __future__ import annotations

import pytest

from plan_and_execute_research_assistant.models import Plan, PlanStep
from plan_and_execute_research_assistant.planning import (
    CircularDependencyError,
    DuplicateStepIdError,
    EmptyPlanError,
    MissingDependencyError,
    PlanValidationError,
    PlanValidator,
    SelfDependencyError,
    validate_plan,
)


def test_valid_linear_plan() -> None:
    """Verify linear dependency chain A -> B -> C validates successfully."""
    plan = Plan(
        goal="Linear Workflow",
        steps=[
            PlanStep(id="s1", description="Step 1", success_criteria="Done"),
            PlanStep(id="s2", description="Step 2", depends_on=["s1"], success_criteria="Done"),
            PlanStep(id="s3", description="Step 3", depends_on=["s2"], success_criteria="Done"),
        ],
    )
    assert validate_plan(plan) is True
    assert PlanValidator.validate(plan) is True


def test_valid_diamond_dag() -> None:
    """Verify diamond fork-join DAG (A -> B, A -> C, B+C -> D) validates successfully."""
    plan = Plan(
        goal="Diamond Workflow",
        steps=[
            PlanStep(id="s_root", description="Root task", success_criteria="Done"),
            PlanStep(
                id="s_branch1",
                description="Parallel Branch 1",
                depends_on=["s_root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="s_branch2",
                description="Parallel Branch 2",
                depends_on=["s_root"],
                success_criteria="Done",
            ),
            PlanStep(
                id="s_join",
                description="Join Branch",
                depends_on=["s_branch1", "s_branch2"],
                success_criteria="Done",
            ),
        ],
    )
    assert validate_plan(plan) is True


def test_valid_multi_root_dag() -> None:
    """Verify multi-root DAG with multiple independent starting tasks."""
    plan = Plan(
        goal="Multi-Root Workflow",
        steps=[
            PlanStep(id="root_a", description="Fetch KB", success_criteria="Done"),
            PlanStep(id="root_b", description="Fetch Account", success_criteria="Done"),
            PlanStep(
                id="merge",
                description="Synthesize",
                depends_on=["root_a", "root_b"],
                success_criteria="Done",
            ),
        ],
    )
    assert validate_plan(plan) is True


def test_empty_plan_error() -> None:
    """Verify EmptyPlanError is raised for zero steps."""
    plan = Plan(goal="Empty", steps=[])
    with pytest.raises(EmptyPlanError, match="Plan must contain at least one step"):
        validate_plan(plan)


def test_empty_step_id_error() -> None:
    """Verify PlanValidationError is raised if a step has an empty ID."""
    plan = Plan(
        goal="Bad ID",
        steps=[PlanStep(id="   ", description="Empty ID step", success_criteria="Done")],
    )
    with pytest.raises(PlanValidationError, match="Step ID cannot be empty"):
        validate_plan(plan)


def test_duplicate_step_id_error() -> None:
    """Verify DuplicateStepIdError is raised when two steps share the same ID."""
    plan = Plan(
        goal="Duplicate ID",
        steps=[
            PlanStep(id="step_dup", description="First instance", success_criteria="Done"),
            PlanStep(id="step_dup", description="Second instance", success_criteria="Done"),
        ],
    )
    with pytest.raises(DuplicateStepIdError, match="Duplicate step ID found: 'step_dup'"):
        validate_plan(plan)


def test_self_dependency_error() -> None:
    """Verify SelfDependencyError is raised when a step depends on itself."""
    plan = Plan(
        goal="Self Dep",
        steps=[
            PlanStep(
                id="step_loop",
                description="Depends on self",
                depends_on=["step_loop"],
                success_criteria="Done",
            )
        ],
    )
    with pytest.raises(SelfDependencyError, match="Step 'step_loop' cannot depend on itself"):
        validate_plan(plan)


def test_missing_dependency_error() -> None:
    """Verify MissingDependencyError is raised when a step references a ghost step."""
    plan = Plan(
        goal="Missing Dep",
        steps=[
            PlanStep(
                id="step_a",
                description="Depends on non-existent",
                depends_on=["ghost_step"],
                success_criteria="Done",
            )
        ],
    )
    with pytest.raises(
        MissingDependencyError, match="Step 'step_a' depends on non-existent step 'ghost_step'"
    ):
        validate_plan(plan)


def test_direct_two_node_cycle() -> None:
    """Verify CircularDependencyError is raised for A -> B -> A."""
    plan = Plan(
        goal="Direct Cycle",
        steps=[
            PlanStep(id="step_a", description="A", depends_on=["step_b"], success_criteria="Done"),
            PlanStep(id="step_b", description="B", depends_on=["step_a"], success_criteria="Done"),
        ],
    )
    with pytest.raises(CircularDependencyError) as exc_info:
        validate_plan(plan)
    assert "Circular dependency detected" in str(exc_info.value)
    assert "step_a" in str(exc_info.value)
    assert "step_b" in str(exc_info.value)


def test_indirect_three_node_cycle() -> None:
    """Verify CircularDependencyError is raised for A -> B -> C -> A."""
    plan = Plan(
        goal="3-Node Cycle",
        steps=[
            PlanStep(id="s1", description="1", depends_on=["s3"], success_criteria="Done"),
            PlanStep(id="s2", description="2", depends_on=["s1"], success_criteria="Done"),
            PlanStep(id="s3", description="3", depends_on=["s2"], success_criteria="Done"),
        ],
    )
    with pytest.raises(CircularDependencyError) as exc_info:
        validate_plan(plan)
    assert "Circular dependency detected" in str(exc_info.value)


def test_cycle_within_subgraph() -> None:
    """Verify cycle detection catches cycles embedded within an otherwise valid DAG."""
    plan = Plan(
        goal="Embedded Cycle",
        steps=[
            PlanStep(id="root", description="Root task", success_criteria="Done"),
            PlanStep(
                id="sub_a",
                description="Sub A",
                depends_on=["root", "sub_b"],
                success_criteria="Done",
            ),
            PlanStep(
                id="sub_b", description="Sub B", depends_on=["sub_a"], success_criteria="Done"
            ),
        ],
    )
    with pytest.raises(CircularDependencyError) as exc_info:
        validate_plan(plan)
    assert "Circular dependency detected" in str(exc_info.value)


def test_unreachable_plan_no_roots() -> None:
    """Verify that a plan where all steps are dependent is rejected as a cycle / unreachable."""
    plan = Plan(
        goal="All Dependent",
        steps=[
            PlanStep(id="s1", description="1", depends_on=["s2"], success_criteria="Done"),
            PlanStep(id="s2", description="2", depends_on=["s1"], success_criteria="Done"),
        ],
    )
    with pytest.raises(CircularDependencyError) as exc_info:
        validate_plan(plan)
    assert "Circular dependency detected" in str(exc_info.value)
