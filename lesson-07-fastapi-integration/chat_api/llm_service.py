"""LLM provider integration."""

from collections.abc import Iterator
from typing import cast

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

from chat_api.config import Settings
from chat_api.session_store import ChatMessage, SessionStore


class LLMService:
    """Provider-isolated LLM chat service."""

    def __init__(
        self,
        store: SessionStore,
        settings: Settings,
    ) -> None:
        self._store = store
        self._settings = settings

        self._client = Groq(
            api_key=settings.groq_api_key
        )

    def _build_provider_messages(
        self,
        system_prompt: str,
        history: list[ChatMessage],
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal history into Groq message format."""

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for message in history:
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    message,
                )
            )

        return messages

    def stream_response(
        self,
        session_id: str,
        user_message: str,
    ) -> Iterator[str]:
        """Yield response chunks and update conversation state."""

        session = self._store.get(
            session_id
        )

        session.add_user_message(
            user_message
        )

        provider_messages = self._build_provider_messages(
            system_prompt=session.system_prompt,
            history=session.messages,
        )

        stream = self._client.chat.completions.create(
            model=self._settings.model_name,
            max_tokens=1024,
            messages=provider_messages,
            stream=True,
        )

        full_text = ""
        final_chunk = None

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                full_text += delta
                yield delta

            final_chunk = chunk

        session.add_assistant_message(
            full_text
        )

        if (
            final_chunk is not None
            and hasattr(final_chunk, "x_groq")
            and final_chunk.x_groq is not None
        ):
            usage = final_chunk.x_groq.usage

            if usage is not None:
                session.total_input_tokens += (
                    usage.prompt_tokens
                )
                session.total_output_tokens += (
                    usage.completion_tokens
                )