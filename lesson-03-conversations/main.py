from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

load_dotenv()

client = Groq()

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
    """A multi-turn chat session with sliding-window context management."""

    system_prompt: str
    max_history_messages: int = 20
    model: str = "openai/gpt-oss-120b"

    _messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0

    def _trim_history(self) -> list[ChatCompletionMessageParam]:
        """Keep only the most recent history messages."""

        if len(self._messages) <= self.max_history_messages:
            return self._messages

        return self._messages[-self.max_history_messages:]

    def _build_messages(self) -> list[ChatCompletionMessageParam]:
        """Build the complete message list sent to the LLM."""

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *self._trim_history(),
        ]

    def send(self, user_message: str) -> str:
        """Send a user message, get a reply, and update conversation state."""

        self._messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=self._build_messages(),
        )

        reply = response.choices[0].message.content

        if reply is None:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        self._messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        usage = response.usage

        if usage is not None:
            self._total_input_tokens += usage.prompt_tokens
            self._total_output_tokens += usage.completion_tokens

        return reply
    
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

        estimated_cost_usd = input_cost + output_cost

        return {
            "turns": len(self._messages) // 2,
            "history_size": len(self._messages),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }

    def history(self) -> list[ChatCompletionMessageParam]:
        """Return the stored conversation history."""

        return self._messages.copy()

    def clear(self) -> None:
        """Reset conversation history and token statistics."""

        self._messages.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0


def main() -> None:
    """Run the command-line chat application."""

    system_prompt = (
    "You are participating in a conversation-memory experiment. "
    "When the user gives you a statement, reply only with 'Noted.' "
    "Do not repeat or summarize information from earlier messages. "
    "When the user asks a question, answer only from the conversation "
    "context available to you. If the answer is not available, say "
    "'I don't know from the available conversation context.'"
    )

    session = ChatSession(
        system_prompt=system_prompt,
        max_history_messages=30,
    )

    print("Chat with the AI. Commands:")
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

            elif user_input == "/stats":
                print(session.stats())
                continue

            elif user_input == "/clear":
                session.clear()
                print("Conversation cleared.")
                continue

            reply = session.send(user_input)
            print(f"\nBot: {reply}")

    except KeyboardInterrupt:
        print("\nExiting.")

    print(f"\nFinal stats: {session.stats()}")


if __name__ == "__main__":
    main()

