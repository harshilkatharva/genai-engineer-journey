from ai_app.core.AIConfig import AiConfig
from ai_app.core.config import DATABASE_CONNECTION_CONVERSATION_URL
from ai_app.db.db_conversation_operations import DBOperator
from ai_app.models.message import Message
from ai_app.models import LLMResponseModel

from uuid import UUID
import re


class ConversationManager:
    def __init__(self):
        self.ai_config = AiConfig()
        self.db_operator = DBOperator(connection_string=DATABASE_CONNECTION_CONVERSATION_URL)

    async def start_conversation(self, user_id: UUID):
        conversation_id = await self.db_operator.create_conversation(user_id)

        return conversation_id

    async def add_conversation(
        self,
        conversation_id,
        user_id,
        request_id,
        role: str,
        content: str,
        feature: str,
        llm_model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        duration_ms: float,
    ) -> None:
        await self.db_operator.add_history(
            conversation_id,
            user_id,
            request_id,
            role,
            content,
            feature,
            llm_model,
            input_tokens,
            output_tokens,
            estimated_cost,
            duration_ms,
        )

    async def get_conversations(self, conversation_id):
        conversations = await self.db_operator.get_history(conversation_id)

        if len(conversations) > 0:
            truncate_conversation = self._truncate_history(conversations)
            return truncate_conversation
        else:
            return conversations

    def _truncate_history(self, history: list[Message]):
        truncated = []
        token_counts = 0

        for message in reversed(history):
            if (
                token_counts + message.input_tokens + message.output_tokens
                > self.ai_config.conversation_history_max_token_size
            ):
                break

            token_counts += message.input_tokens + message.output_tokens
            truncated.append(message)

        return truncated

    def _normalize_text(text: str) -> str:
        """
        Normalize user input for exact-match caching.

        Examples:
            "  Summarize this  " -> "summarize this"
            "SUMMARIZE   THIS"  -> "summarize this"
        """
        return re.sub(r"\s+", " ", text.strip().lower())

    async def get_cached_response(
        self,
        user_message: str,
        feature: str = "summarization",
    ):
        normalized_message = self._normalize_text(user_message)

        cached_history = await self.db_operator.get_cached_history(
            normalized_message=normalized_message,
            feature=feature,
        )

        if cached_history is None or len(cached_history) == 0:
            return None

        (
            content,
            llm_model,
            input_tokens,
            output_tokens,
            duration_ms,
        ) = cached_history

        return LLMResponseModel(
            text=content,
            model=llm_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )
