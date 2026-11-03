import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

MODEL = "openai/gpt-oss-120b"


def stream_and_measure(prompt: str) -> dict:
    """Stream a response, measure latency, and return metrics."""

    start = time.perf_counter()
    first_token_at = None

    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )

    full_text = ""
    final_chunk = None

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()

            print(
                delta,
                end="",
                flush=True,
            )

            full_text += delta

        final_chunk = chunk

    end = time.perf_counter()

    print()

    if first_token_at is not None:
        time_to_first_token = first_token_at - start
    else:
        time_to_first_token = None

    total_time = end - start

    total_tokens = None

    if (
        final_chunk is not None
        and hasattr(final_chunk, "x_groq")
        and final_chunk.x_groq is not None
    ):
        usage = final_chunk.x_groq.usage

        if usage is not None:
            total_tokens = usage.total_tokens

    return {
        "time_to_first_token": time_to_first_token,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "full_text": full_text,
    }


def no_stream_and_measure(prompt: str) -> dict:
    """Send a non-streaming request and measure total latency."""

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    end = time.perf_counter()

    total_time = end - start

    full_text = response.choices[0].message.content

    if full_text is None:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    print(full_text)

    usage = response.usage

    total_tokens = (
        usage.total_tokens
        if usage is not None
        else None
    )

    return {
        "time_to_first_token": total_time,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "full_text": full_text,
    }
    

def main() -> None:
    """Run streaming and non-streaming timing experiments."""

    long_prompt = (
        "Explain how B-tree indexes work in Postgres. "
        "Include an example with a specific table structure. "
        "Aim for a thorough answer with 5-6 paragraphs."
    )

    print("\n" + "=" * 60)
    print("Experiment A — Streaming (long prompt)")
    print("=" * 60)

    result_a = stream_and_measure(long_prompt)

    print("\n--- Metrics ---")
    print(
        f"Time to first token: "
        f"{result_a['time_to_first_token']:.2f}s"
    )
    print(
        f"Total time: "
        f"{result_a['total_time']:.2f}s"
    )
    print(
        f"Total tokens: "
        f"{result_a['total_tokens']}"
    )
    print(
        f"Full text length: "
        f"{len(result_a['full_text'])} characters"
    )

    print("\n" + "=" * 60)
    print("Experiment B — Non-streaming (same long prompt)")
    print("=" * 60)

    result_b = no_stream_and_measure(long_prompt)

    print("\n--- Metrics ---")
    print(
        f"Time to first token: "
        f"{result_b['time_to_first_token']:.2f}s"
    )
    print(
        f"Total time: "
        f"{result_b['total_time']:.2f}s"
    )
    print(
        f"Total tokens: "
        f"{result_b['total_tokens']}"
    )
    print(
        f"Full text length: "
        f"{len(result_b['full_text'])} characters"
    )

    print("\n" + "=" * 60)
    print("Experiment C — Streaming (short prompt)")
    print("=" * 60)

    result_c = stream_and_measure(
        "What is 2+2?"
    )

    print("\n--- Metrics ---")
    print(
        f"Time to first token: "
        f"{result_c['time_to_first_token']:.2f}s"
    )
    print(
        f"Total time: "
        f"{result_c['total_time']:.2f}s"
    )
    print(
        f"Total tokens: "
        f"{result_c['total_tokens']}"
    )
    print(
        f"Full text length: "
        f"{len(result_c['full_text'])} characters"
    )


if __name__ == "__main__":
    main()