"""Unit tests verifying integration test gating behavior."""

from __future__ import annotations

import pytest

from tests.conftest import check_integration_gate


def test_require_integration_test_flag_skips_when_disabled() -> None:
    """When marker is integration and INTEGRATION_TEST is not '1', it should skip."""
    keywords = {"integration": True}

    with pytest.raises(pytest.skip.Exception, match="Integration tests are disabled"):
        check_integration_gate(keywords, None)

    with pytest.raises(pytest.skip.Exception, match="Integration tests are disabled"):
        check_integration_gate(keywords, "0")


def test_require_integration_test_flag_allows_when_enabled() -> None:
    """When marker is integration and INTEGRATION_TEST is '1', it should pass."""
    keywords = {"integration": True}

    # Should not raise pytest.skip
    check_integration_gate(keywords, "1")


def test_require_integration_test_flag_ignores_non_integration_tests() -> None:
    """Non-integration tests should execute regardless of INTEGRATION_TEST."""
    keywords = {"unit": True}

    # Should not raise pytest.skip
    check_integration_gate(keywords, "0")
    check_integration_gate(keywords, None)
