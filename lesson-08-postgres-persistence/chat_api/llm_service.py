

"""LLM orchestration with PostgreSQL persistence.

Provider-specific code is isolated here.

For Anthropic, the Groq client initialization, message format,
streaming API, and usage extraction would change.

The repository and database models would remain unchanged.
"""

from collections.abc import Iterator
from typing import cast

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from sqlalchemy.orm import Session, sessionmaker

from chat_api import repository
from chat_api.config import Settings
from chat_api.models import ChatMessage


class LLMService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._client = Groq(api_key=settings.groq_api_key)

    def stream_response(
        self,
        public_session_id: str,
        user_message: str,
    ) -> Iterator[str]:
        """Stream LLM output and persist the conversation."""

        # STEP 1: Load the conversation and save the user's message.
        with self._session_factory() as db:
            chat_session = repository.get_session_by_public_id(
                db,
                public_session_id,
            )

            internal_session_id = chat_session.id

            # Build the history before adding the new message.
            # This avoids relying on a stale ORM relationship.
            provider_messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": chat_session.system_prompt,
                }
            ]

            for message in chat_session.messages:
                provider_messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {
                            "role": message.role,
                            "content": message.content,
                        },
                    )
                )

            repository.append_message(
                db=db,
                session_id=internal_session_id,
                role="user",
                content=user_message,
            )

            db.commit()

        # STEP 2: Include the current user's message.
        provider_messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # STEP 3: Call Groq and stream the response.
        stream = self._client.chat.completions.create(
            model=self._settings.model_name,
            max_tokens=1024,
            messages=provider_messages,
            stream=True,            
        )

        full_text = ""
        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content

                if delta:
                    full_text += delta
                    yield delta

            # Groq may return usage information in a final chunk.
            usage = getattr(chunk, "x_groq", None)

            if usage is not None and usage.usage is not None:
                input_tokens = usage.usage.prompt_tokens
                output_tokens = usage.usage.completion_tokens

        # STEP 4: Persist the complete assistant response.
        with self._session_factory() as db:
            repository.append_message(
                db=db,
                session_id=internal_session_id,
                role="assistant",
                content=full_text,
            )

            repository.update_session_tokens(
                db=db,
                session_id=internal_session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            db.commit()


    def _build_provider_messages(
        self,
        system_prompt: str,
        history: list[ChatMessage],
    ) -> list[ChatCompletionMessageParam]:
        """Convert database messages into Groq's message format."""

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
                    {
                        "role": message.role,
                        "content": message.content,
                    },
                )
            )

        return messages