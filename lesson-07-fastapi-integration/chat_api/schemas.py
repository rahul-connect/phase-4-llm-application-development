"""Pydantic schemas for the chat API."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Optional settings when creating a new chat session."""

    system_prompt: str | None = None


class CreateSessionResponse(BaseModel):
    """Response returned after creating a chat session."""

    session_id: str
    created_at: datetime


class MessageRequest(BaseModel):
    """A user message sent to an existing chat session."""

    message: str = Field(
        min_length=1,
        description="The user's chat message.",
    )