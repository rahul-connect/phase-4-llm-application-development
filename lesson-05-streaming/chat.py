"""Streaming ChatSession — Lesson 4.5."""

from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam


load_dotenv()

client = Groq()

MODEL = "openai/gpt-oss-120b"

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

@dataclass
class ChatSession:
    """A multi-turn chat session with streaming responses."""

    system_prompt: str
    max_history_messages: int = 20
    model: str = MODEL

    _messages: list[ChatCompletionMessageParam] = field(
        default_factory=list
    )
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0

    def _trim_history(self) -> list[ChatCompletionMessageParam]:
        """Keep only the most recent history messages."""

        if len(self._messages) <= self.max_history_messages:
            return self._messages

        return self._messages[-self.max_history_messages:]

    def _build_messages(self) -> list[ChatCompletionMessageParam]:
        """Build the full messages payload for the LLM."""

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *self._trim_history(),
        ]

    def send(self, user_message: str) -> str:
        """Send a message, stream the reply, and update session state."""

        self._messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        stream = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=self._build_messages(),
            stream=True,
        )

        full_text = ""
        final_chunk = None

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                print(
                    delta,
                    end="",
                    flush=True,
                )

                full_text += delta

            final_chunk = chunk

        print()

        self._messages.append(
            {
                "role": "assistant",
                "content": full_text,
            }
        )

        if (
            final_chunk is not None
            and hasattr(final_chunk, "x_groq")
            and final_chunk.x_groq is not None
        ):
            usage = final_chunk.x_groq.usage

            if usage is not None:
                self._total_input_tokens += usage.prompt_tokens
                self._total_output_tokens += usage.completion_tokens

        return full_text


    def stats(self) -> dict:
        """Return conversation statistics."""

        prices = GROQ_PRICES[self.model]

        input_cost = (
            self._total_input_tokens
            * prices["input"]
        )

        output_cost = (
            self._total_output_tokens
            * prices["output"]
        )

        return {
            "turns": len(self._messages) // 2,
            "history_size": len(self._messages),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": input_cost + output_cost,
        }

    def history(self) -> list[ChatCompletionMessageParam]:
        """Return a copy of the stored conversation history."""

        return self._messages.copy()

    def clear(self) -> None:
        """Clear conversation history and token statistics."""

        self._messages.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0


def main() -> None:
    """Run an interactive streaming chat."""

    system_prompt = (
    "You are an experienced Python and backend engineering tutor. "
    "Answer clearly, concisely, and no need to provide code examples or markdown tables."
)

    session = ChatSession(
        system_prompt=system_prompt,
        max_history_messages=20,
    )

    print("Streaming chat. Commands:")
    print("/stats  — show conversation statistics")
    print("/clear  — clear conversation history")
    print("/quit   — exit")
    print("-" * 60)

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                break

            if user_input == "/stats":
                print(session.stats())
                continue

            if user_input == "/clear":
                session.clear()
                print("Conversation cleared.")
                continue

            print("\nBot: ", end="", flush=True)
            session.send(user_input)

    except KeyboardInterrupt:
        print("\nExiting.")

    print(f"\nFinal stats: {session.stats()}")


if __name__ == "__main__":
    main()