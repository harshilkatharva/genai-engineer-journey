import os

import pytest


def pytest_collection_modifyitems(config, items):
    env_names = ("INTEGRATION_TEST", "integrGRATION_TEST")
    enabled = any(os.getenv(name) == "1" for name in env_names)

    if enabled:
        return

    skip_integration = pytest.mark.skip(
        reason="integration tests require INTEGRATION_TEST=1 or integrGRATION_TEST=1 in the environment",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
