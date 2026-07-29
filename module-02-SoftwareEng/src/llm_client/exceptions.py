class LLMError(Exception):
    """
    Base Exception for all error
    """

    def __init__(
        self, message: str = "An unknown LLM error occured", user_message: str | None = None
    ):
        self.message = message
        super().__init__(self.message)
        self.user_message = user_message or message


class ConfigError(LLMError):
    def __init__(self, key: str):
        self.key = key
        self.message = f"Missing required configuration: '{key}'. Please set it in your .env file."


class LLMRateLimitError(LLMError):
    """
    This error occured when provider reject request becuase the rate limit has been exceeded
    """

    def __init__(self, message: str):
        super().__init__(
            message=message, user_message="Rate limit exceeded. Please Try again later."
        )


class LLMTimeoutError(LLMError):
    """
    Raised when an API request exceeds the configured timeout.
    """

    def __init__(self, message: str):
        super().__init__(
            message=message,
            user_message="Timeout due to long waiting time. Please Try again later.",
        )


class LLMContentFilterError(LLMError):
    """
    Raised when the provider blocks the prompt or response due to safety/content filtering.
    """

    def __init__(self, message: str):
        super().__init__(
            message=message,
            user_message="Your message contain Invalid content. Please modify your message before retry.",
        )


class LLMInvalidResponseError(LLMError):
    """
    Raised when the provider returns an unexpected or invalid response format.
    """

    def __init__(self, message: str):
        super().__init__(message=message, user_message="Invalid response from provider.")


class LLMAuthenticationError(LLMError):
    """
    Raised when API authentication fails due to an invalid or missing API key.
    """

    def __init__(self, message: str):
        super().__init__(message=message, user_message="You are unauthorized to access provider.")


class LLMConnectionError(LLMError):
    """
    Raised when the client cannot connect to the provider because of network or connectivity issues.
    """

    def __init__(self, message: str):
        super().__init__(
            message=message,
            user_message="We can not connect with server. Please try again after some time.",
        )
