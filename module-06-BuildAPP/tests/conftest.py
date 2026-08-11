from unittest.mock import patch

import pytest

from ai_app.core.conversation_manager import ConversationManager


@pytest.fixture(autouse=False, scope="session")
def conversation_manager():
    with (
        patch("ai_app.core.conversation_manager.AiConfig") as mock_ai_config,
    ):
        mock_ai_config.return_value.conversation_history_max_token_size = 100

        conversation_manager = ConversationManager()

        yield conversation_manager
