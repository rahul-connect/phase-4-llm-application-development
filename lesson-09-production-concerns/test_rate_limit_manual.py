from chat_api.rate_limit import RateLimiter


limiter = RateLimiter(
    capacity=3,
    rate=1 / 60,
)

session_id = "test-session"

for i in range(5):

    allowed = limiter.check(session_id)

    print(f"Request {i + 1}: {allowed}")