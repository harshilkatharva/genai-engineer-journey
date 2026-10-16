class LLMError(Exception):
    """Base exception for LLM operations."""

    def __init__(
        self,
        message: str = "An unknown LLM error occurred",
        user_message: str | None = None,
    ):
        self.message = message
        super().__init__(self.message)
        self.user_message = user_message or message


class ConfigError(LLMError):
    def __init__(self, key: str):
        self.key = key
        message = f"Missing required configuration: '{key}'. Please set it in your .env file."
        super().__init__(message)


class LLMRateLimitError(LLMError):
    def __init__(self, message: str):
        super().__init__(
            message=message, user_message="Rate limit exceeded. Please try again later."
        )


class LLMTimeoutError(LLMError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            user_message="Timeout due to long waiting time. Please try again later.",
        )


class LLMContentFilterError(LLMError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            user_message="Your message contains invalid content. Please modify your message before retry.",
        )


class LLMInvalidResponseError(LLMError):
    def __init__(self, message: str):
        super().__init__(message=message, user_message="Invalid response from provider.")


class LLMAuthenticationError(LLMError):
    def __init__(self, message: str):
        super().__init__(message=message, user_message="You are unauthorized to access provider.")


class LLMConnectionError(LLMError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            user_message="Cannot connect with server. Please try again later.",
        )


class LLMValidationError(LLMError):
    """Raised when LLM output cannot be parsed as JSON or fails Pydantic schema validation after retries."""

    def __init__(
        self,
        message: str,
        raw_output: str | None = None,
        errors: list[object] | None = None,
    ):
        super().__init__(
            message=message,
            user_message="LLM output failed schema validation.",
        )
        self.raw_output = raw_output
        self.errors = errors or []
