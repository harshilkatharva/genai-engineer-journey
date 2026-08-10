from pydantic_settings import BaseSettings


class AiConfig(BaseSettings):
    default_prvoider = "google"
    default_model: str = "gemini-3.5-flash-lite"
    fallback_provider = "google"
    fallback_model: str = "gemini-3.5-flash"
    default_temprature: float = 0.3
    conversation_history_max_token_size: int = 6000
    enable_streaming: bool = True
    feature_flag: dict[str, bool] = {"enable_rag": False, "enable_agents": False}
