from chat_api.retry import retry_with_backoff


attempts = 0


def simulate_provider():
    global attempts

    attempts += 1

    print(f"Attempt {attempts}")

    if attempts < 3:
        raise ConnectionError("Simulated connection failure")

    return "Provider connected"


try:
    result = retry_with_backoff(
        func=simulate_provider,
        max_attempts=3,
        base_delay=0.1,
    )

    print(result)

except Exception as exc:
    print(f"Final error: {exc}")