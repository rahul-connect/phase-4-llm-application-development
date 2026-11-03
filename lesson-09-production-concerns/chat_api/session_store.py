"""In-memory chat session storage."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, TypedDict
from uuid import uuid4

from chat_api.exceptions import SessionNotFoundError


class ChatMessage(TypedDict):
    """Provider-neutral chat message."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class ChatSession:
    """One in-memory chat conversation."""

    session_id: str
    system_prompt: str
    created_at: datetime
    messages: list[ChatMessage] = field(
        default_factory=list
    )
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_user_message(
        self,
        content: str,
    ) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )


class SessionStore:
    """In-memory storage for chat sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(
        self,
        system_prompt: str,
    ) -> ChatSession:
        session_id = str(uuid4())

        session = ChatSession(
            session_id=session_id,
            system_prompt=system_prompt,
            created_at=datetime.now(timezone.utc),
        )

        self._sessions[session_id] = session

        return session

    def get(
        self,
        session_id: str,
    ) -> ChatSession:
        session = self._sessions.get(
            session_id
        )

        if session is None:
            raise SessionNotFoundError(
                session_id
            )

        return session

    def exists(
        self,
        session_id: str,
    ) -> bool:
        return session_id in self._sessions