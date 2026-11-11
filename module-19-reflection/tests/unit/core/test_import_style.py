"""Regression test enforcing standardized absolute imports across the package."""

from __future__ import annotations

import ast
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SEARCH_ROOTS = (
    _PROJECT_ROOT / "src" / "self_improving_report_writer",
    _PROJECT_ROOT / "tests",
)


def test_no_relative_imports_in_package_or_tests() -> None:
    """All intra-package imports must use explicit absolute `self_improving_report_writer` paths."""
    violations: list[str] = []
    for root in _SEARCH_ROOTS:
        for py_file in sorted(root.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    rel_path = py_file.relative_to(_PROJECT_ROOT)
                    module_suffix = node.module or ""
                    violations.append(
                        f"{rel_path}:{node.lineno} (from {'.' * node.level}{module_suffix})"
                    )

    assert not violations, "Relative imports detected:\n" + "\n".join(violations)
