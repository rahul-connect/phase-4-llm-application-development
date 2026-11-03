import random
import time

from collections.abc import Callable
from typing import TypeVar

from groq import RateLimitError

from chat_api.errors import LLMErrorCategory, categorize_error


T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
) -> T:

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(max_attempts):

        try:
            return func()

        except Exception as exc:

            category = categorize_error(exc)

            if category != LLMErrorCategory.TRANSIENT:
                raise

            if attempt == max_attempts - 1:
                raise

            delay = min(
                base_delay * (2 ** attempt),
                max_delay,
            )

            delay += random.uniform(0, delay * jitter)

            if isinstance(exc, RateLimitError):
                retry_after = exc.response.headers.get("retry-after")

                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass

            time.sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")