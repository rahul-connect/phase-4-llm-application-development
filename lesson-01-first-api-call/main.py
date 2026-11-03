"""Explore the Groq API and track token usage and estimated cost."""

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq()


# These prices come from the lesson's pricing assumptions.
# Pricing can change, so in production we would keep this updated
# from the provider's official pricing documentation.
GROQ_PRICES = {
    "openai/gpt-oss-120b": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "openai/gpt-oss-20b": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000,
    },
}


def ask_llm(
    prompt: str,
    system: str | None = None,
    model: str = "openai/gpt-oss-120b",
) -> dict:
    """Send a prompt to Groq and return text, usage, cost, and metadata."""

    messages = []

    if system is not None:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=messages,
    )

    # The SDK types allow usage to possibly be None.
    # For this exercise we require usage because we want to calculate cost.
    usage = response.usage

    if usage is None:
        raise RuntimeError(
            "Groq response did not include token usage information."
        )

    text = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    actual_model = response.model

    prices = GROQ_PRICES[model]

    input_cost = (
        prompt_tokens
        * prices["input"]
    )

    output_cost = (
        completion_tokens
        * prices["output"]
    )

    cost_usd = input_cost + output_cost

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "finish_reason": finish_reason,
        "model": actual_model,
    }


def print_result(
    label: str,
    result: dict,
) -> None:
    """Print one experiment result in a readable format."""

    print(f"\n--- Experiment {label} ---")
    print(f"Model: {result['model']}")
    print(f"Text:\n{result['text']}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Completion tokens: {result['completion_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Finish reason: {result['finish_reason']}")
    print(f"Estimated cost: ${result['cost_usd']:.8f}")


def main() -> None:
    """Run the three Groq experiments."""

    result_a = ask_llm(
        "What is Postgres in one sentence?"
    )

    result_b = ask_llm(
        "What is Postgres in one sentence?",
        system=(
            "Respond only in JSON with a 'summary' field."
        ),
    )

    result_c = ask_llm(
    "Write a haiku about databases.",
    model="openai/gpt-oss-20b",
    )

    print_result("A", result_a)
    print_result("B", result_b)
    print_result("C", result_c)

    total_cost = (
        result_a["cost_usd"]
        + result_b["cost_usd"]
        + result_c["cost_usd"]
    )

    print("\n--- Total ---")
    print(
        f"Estimated cost across all three calls: "
        f"${total_cost:.8f}"
    )


if __name__ == "__main__":
    main()