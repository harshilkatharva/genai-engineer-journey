class LLMError(Exception):
    """
    Base Exception for all error
    """

    def __ini__(self, message: str = "An unknowm=n LLM error occured"):
        self.message = message
        super().__init__(self.message)


class LLMRateLimitError(LLMError):
    """
    This error occured when provider reject request becuase the rate limit has been exceeded
    """


class LLMTimeoutError(LLMError):
    """
    Raised when an API request exceeds the configured timeout.
    """


class LLMContentFilterError(LLMError):
    """
    Raised when the provider blocks the prompt or response due to safety/content filtering.
    """


class LLMInvalidResponseError(LLMError):
    """
    Raised when the provider returns an unexpected or invalid response format.
    """


class LLMAuthenticationError(LLMError):
    """
    Raised when API authentication fails due to an invalid or missing API key.
    """


class LLMConnectionError(LLMError):
    """
    Raised when the client cannot connect to the provider because of network or connectivity issues.
    """
