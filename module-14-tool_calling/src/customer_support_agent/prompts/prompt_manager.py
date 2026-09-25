from customer_support_agent.models import ChatMessage


class SupportPromptManager:
    """Builds the system instruction used by the Phase 2 agent loop."""

    SYSTEM_PROMPT = """You are an e-commerce customer support agent.
Answer using only the conversation and results returned by the available tools.
Use a tool when the user asks for order, product, policy, or escalation information.
Never invent order, product, policy, or ticket details. Be concise and transparent."""

    def build_messages(self, user_query: str) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_query),
        ]
