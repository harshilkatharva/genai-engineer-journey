import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.getenv("INTEGRATION_TEST") == "1":
        return

    skip_integration = pytest.mark.skip(reason="Set INTEGRATION_TEST=1 to run integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
