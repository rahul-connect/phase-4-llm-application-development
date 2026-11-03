from dataclasses import dataclass, field
from time import monotonic


@dataclass
class TokenBucket:
    capacity: int
    rate: float

    tokens: float = 0.0

    last_refill: float = field(default_factory=monotonic)

    def try_consume(self, amount: int = 1) -> bool:

        now = monotonic()

        elapsed = now - self.last_refill

        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate,
        )

        self.last_refill = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True

        return False


class RateLimiter:

    def __init__(self, capacity: int, rate: float):

        self._buckets: dict[str, TokenBucket] = {}

        self._capacity = capacity
        self._rate = rate

    def check(self, key: str) -> bool:

        if key not in self._buckets:

            self._buckets[key] = TokenBucket(
                capacity=self._capacity,
                rate=self._rate,
                tokens=self._capacity,
            )

        return self._buckets[key].try_consume()



chat_rate_limiter = RateLimiter(
    capacity=10,
    rate=10 / 60,
)