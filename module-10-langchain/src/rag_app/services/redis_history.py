from langchain_redis import RedisChatMessageHistory

from rag_app.core.settings import get_settings


def get_redis_message_history(session_id: str) -> RedisChatMessageHistory:
    settings = get_settings()

    return RedisChatMessageHistory(
        session_id=session_id,
        redis_url=settings.redis_url,
        key_prefix="rag:chat:",
        index_name="idx:rag_chat_history",
        overwrite_index=True,
    )
