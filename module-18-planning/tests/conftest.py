"""Shared test configuration and fixtures for Module 18."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Conditionally skip integration tests unless INTEGRATION_TEST=1 is configured."""
    flag = os.getenv("INTEGRATION_TEST", "").strip().lower()
    run_integration = flag in ("1", "true", "yes", "on")

    skip_integration = pytest.mark.skip(
        reason="Integration tests skipped. Set INTEGRATION_TEST=1 in .env to run them."
    )
    for item in items:
        item_path_str = str(item.fspath)
        is_integration = "/integration" in item_path_str or "integration" in item.keywords
        if is_integration and not run_integration:
            item.add_marker(skip_integration)
