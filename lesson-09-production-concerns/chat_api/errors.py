from enum import Enum

from groq import (
    APIConnectionError,
    APIStatusError,
    RateLimitError,
)


class LLMErrorCategory(str, Enum):
    CORRECTABLE = "correctable"
    TRANSIENT = "transient"
    TERMINAL = "terminal"



def categorize_error(exc: Exception) -> LLMErrorCategory:

    if isinstance(exc, RateLimitError):
        return LLMErrorCategory.TRANSIENT

    if isinstance(exc, APIConnectionError):
        return LLMErrorCategory.TRANSIENT

    if isinstance(exc, APIStatusError):

        if exc.status_code and exc.status_code >= 500:
            return LLMErrorCategory.TRANSIENT

        if exc.status_code in (401, 403):
            return LLMErrorCategory.TERMINAL

        if exc.status_code in (400, 404, 422):
            return LLMErrorCategory.TERMINAL

    return LLMErrorCategory.TERMINAL