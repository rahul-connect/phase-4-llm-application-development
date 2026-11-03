"""FastAPI routes for the chat API."""

import json

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from chat_api.config import Settings, get_settings
from chat_api.llm_service import LLMService
from chat_api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    MessageRequest,
)
from chat_api.session_store import SessionStore


router = APIRouter()

session_store = SessionStore()


def get_session_store() -> SessionStore:
    """Return the shared in-memory session store."""

    return session_store


def get_llm_service(
    store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> LLMService:
    """Build the LLM service from application dependencies."""

    return LLMService(
        store=store,
        settings=settings,
    )


@router.get("/")
def root(
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": settings.app_name,
        "version": "0.1.0",
    }


@router.post(
    "/chat/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    request: CreateSessionRequest,
    store: SessionStore = Depends(get_session_store),
) -> CreateSessionResponse:
    """Create a new chat session."""

    system_prompt = (
        request.system_prompt
        or "You are a helpful assistant."
    )

    session = store.create(
        system_prompt=system_prompt,
    )

    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
    )


@router.post(
    "/chat/sessions/{session_id}/messages",
)
def send_message(
    session_id: str,
    request: MessageRequest,
    store: SessionStore = Depends(get_session_store),
    llm_service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    """Send a message and stream the assistant response."""

    # Validate before streaming starts so a missing session
    # becomes a normal HTTP 404 instead of an SSE error.
    store.get(session_id)

    def event_stream():
        for delta in llm_service.stream_response(
            session_id=session_id,
            user_message=request.message,
        ):
            payload = {
                "delta": delta,
            }

            yield (
                f"data: {json.dumps(payload)}\n\n"
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )