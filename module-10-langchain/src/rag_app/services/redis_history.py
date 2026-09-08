from collections.abc import Sequence

from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import BaseMessage

from rag_app.core.settings import get_settings


def _message_text(content: str | Sequence[object]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])

    return "".join(parts)


class RedisChatHistory(RedisChatMessageHistory):
    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message.content, (list, tuple)):
            message = message.model_copy(update={"content": _message_text(message.content)})

        super().add_message(message)


def get_redis_message_history(session_id: str) -> RedisChatHistory:
    settings = get_settings()

    return RedisChatHistory(
        session_id=session_id,
        redis_url=settings.redis_url,
        key_prefix="rag:chat:",
        index_name="idx:rag_chat_history",
        overwrite_index=True,
    )
