from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        min_length=1,
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime


class SendMessageRequest(BaseModel):
    message: str = Field(
        min_length=1,
    )


class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class SessionDetailResponse(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    total_input_tokens: int
    total_output_tokens: int
    messages: list[MessageResponse]