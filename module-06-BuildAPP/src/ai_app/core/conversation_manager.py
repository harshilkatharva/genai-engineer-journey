from dataclasses import dataclass

from ai_app.core.AIConfig import AiConfig
from ai_app.core.cost_tracker import CostTracker
from ai_app.db.db_conversation_operations import DBOperator


@dataclass
class Message:
    role: str
    content: str
    input_tokens: int
    output_tokens: int


class ConversationManager:
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.ai_config = AiConfig()
        self.db_operator = DBOperator()

    def start_conversation(self, user_id: str):
        conversation_id = self.db_operator.create_conversation(user_id)

        return conversation_id

    def add_conversation(
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
        self.db_operator.add_history(
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

    def get_conversations(self, conversation_id):
        conversations = self.db_operator.get_history(conversation_id)

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
                > self.conversation_history_max_token_size
            ):
                break

            token_counts += message.input_tokens + message.output_tokens
            truncated.append(message)

        return truncated
