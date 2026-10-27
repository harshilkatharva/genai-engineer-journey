"""Structural validator for execution plans."""

from __future__ import annotations

from ..models.plan import Plan, PlanStep
from .exceptions import (
    CircularDependencyError,
    DuplicateStepIdError,
    EmptyPlanError,
    MissingDependencyError,
    PlanValidationError,
    SelfDependencyError,
    UnreachablePlanError,
)

# 3-State DFS node markers for cycle detection
_UNVISITED = 0
_VISITING = 1
_VISITED = 2


class PlanValidator:
    """Validates structural DAG integrity of an execution plan."""

    @classmethod
    def validate(cls, plan: Plan) -> bool:
        """Validate that a Plan satisfies all structural integrity rules.

        Enforces:
          1. Plan has at least one step.
          2. Step IDs are non-empty and unique.
          3. No step depends on itself.
          4. All referenced dependencies exist in the plan.
          5. Dependency graph contains zero cycles (is a valid DAG).
          6. Plan has at least one root step with no dependencies.

        Args:
            plan: The Plan instance to validate.

        Returns:
            bool: True if the plan is completely valid.

        Raises:
            EmptyPlanError: If the plan contains zero steps.
            DuplicateStepIdError: If duplicate step IDs exist.
            SelfDependencyError: If a step depends on itself.
            MissingDependencyError: If a step references a missing dependency.
            CircularDependencyError: If a dependency cycle is detected.
            UnreachablePlanError: If no step has empty dependencies.
            PlanValidationError: For empty or invalid step identifiers.
        """
        # 1. Non-empty plan check
        if not plan.steps:
            raise EmptyPlanError("Plan must contain at least one step.")

        # 2. Check for empty IDs and duplicate IDs
        seen_ids: set[str] = set()
        step_map: dict[str, PlanStep] = {}
        for step in plan.steps:
            if not step.id or not step.id.strip():
                raise PlanValidationError("Step ID cannot be empty.")
            if step.id in seen_ids:
                raise DuplicateStepIdError(f"Duplicate step ID found: '{step.id}'")
            seen_ids.add(step.id)
            step_map[step.id] = step

        # 3. Check for self-dependencies and missing dependencies
        for step in plan.steps:
            deps = step.depends_on or []
            for dep_id in deps:
                if dep_id == step.id:
                    raise SelfDependencyError(f"Step '{step.id}' cannot depend on itself.")
                if dep_id not in seen_ids:
                    raise MissingDependencyError(
                        f"Step '{step.id}' depends on non-existent step '{dep_id}'."
                    )

        # 4. Cycle detection using 3-state Depth-First Search (DFS)
        state: dict[str, int] = {s_id: _UNVISITED for s_id in seen_ids}

        def dfs(node: str, path: list[str]) -> None:
            state[node] = _VISITING
            path.append(node)

            deps = step_map[node].depends_on or []
            for dep in deps:
                if state[dep] == _VISITING:
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    raise CircularDependencyError(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )
                elif state[dep] == _UNVISITED:
                    dfs(dep, path)

            path.pop()
            state[node] = _VISITED

        for step_id in seen_ids:
            if state[step_id] == _UNVISITED:
                dfs(step_id, [])

        # 5. Reachability / root step check: at least one step must have no dependencies
        has_root = any(not s.depends_on for s in plan.steps)
        if not has_root:
            raise UnreachablePlanError(
                "Plan has no entry step with empty dependencies; execution cannot begin."
            )

        return True


def validate_plan(plan: Plan) -> bool:
    """Convenience functional wrapper around PlanValidator.validate."""
    return PlanValidator.validate(plan)
