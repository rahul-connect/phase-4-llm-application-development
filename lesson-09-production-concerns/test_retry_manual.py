from chat_api.retry import retry_with_backoff


def successful_function():
    print("Function executed")
    return "Success!"


result = retry_with_backoff(successful_function)

print(result)