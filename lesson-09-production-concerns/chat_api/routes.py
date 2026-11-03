import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from chat_api import repository
from chat_api.config import get_settings
from chat_api.database import SessionLocal, get_db
from chat_api.llm_service import LLMService
from chat_api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    MessageResponse,
    SendMessageRequest,
    SessionDetailResponse,
)
from chat_api.rate_limit import chat_rate_limiter
from chat_api.errors import LLMErrorCategory, categorize_error

router = APIRouter()


def get_llm_service() -> LLMService:
    """Create the LLM service with its dependencies."""

    return LLMService(
        session_factory=SessionLocal,
        settings=get_settings(),
    )


@router.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Chat API is running",
    }


@router.post(
    "/chat/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
)
def create_chat_session(
    request: CreateSessionRequest,
    db: Session = Depends(get_db),
) -> CreateSessionResponse:

    chat_session = repository.create_session(
        db=db,
        system_prompt=request.system_prompt,
    )

    return CreateSessionResponse(
        session_id=chat_session.public_id,
        created_at=chat_session.created_at,
    )


@router.get(
    "/chat/sessions/{session_id}",
    response_model=SessionDetailResponse,
)
def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> SessionDetailResponse:

    chat_session = repository.get_session_by_public_id(
        db=db,
        public_id=session_id,
    )

    return SessionDetailResponse(
        session_id=chat_session.public_id,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        total_input_tokens=chat_session.total_input_tokens,
        total_output_tokens=chat_session.total_output_tokens,
        messages=[
            MessageResponse(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in chat_session.messages
        ],
    )




@router.post(
    "/chat/sessions/{session_id}/messages",
)
def send_message(
    session_id: str,
    request: SendMessageRequest,
    db: Session = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:

    # STEP 1: Validate session.
    repository.get_session_by_public_id(
        db=db,
        public_id=session_id,
    )

    db.rollback()

    # STEP 2: Check rate limit.
    if not chat_rate_limiter.check(session_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    # STEP 3: Define streaming generator.
    def event_stream() -> Iterator[str]:
        try:
            for delta in llm_service.stream_response(
                public_session_id=session_id,
                user_message=request.message,
            ):
                payload = json.dumps(
                    {"delta": delta},
                    ensure_ascii=False,
                )

                yield f"data: {payload}\n\n"

        except Exception as exc:
            category = categorize_error(exc)

            if category == LLMErrorCategory.TRANSIENT:
                message = "The AI provider is temporarily unavailable."

            elif category == LLMErrorCategory.CORRECTABLE:
                message = "The AI request could not be processed."

            else:
                message = "The AI request failed."

            payload = json.dumps(
                {
                    "error": message,
                    "category": category.value,
                }
            )

            yield f"event: error\ndata: {payload}\n\n"
            return

        yield "data: [DONE]\n\n"
        
    # STEP 4: Return streaming response.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )