"""Database operations for chat sessions and messages."""

from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from chat_api.exceptions import SessionNotFoundError
from chat_api.models import ChatMessage, ChatSession

from decimal import Decimal

from chat_api.models import LLMCall

GROQ_PRICES = {
    "openai/gpt-oss-120b": {
        "input": Decimal("0.15") / 1_000_000,
        "output": Decimal("0.60") / 1_000_000,
    },
    "openai/gpt-oss-20b": {
        "input": Decimal("0.075") / 1_000_000,
        "output": Decimal("0.30") / 1_000_000,
    },
}


def create_session(
    db: Session,
    system_prompt: str,
) -> ChatSession:
    """Create and persist a new chat session."""

    chat_session = ChatSession(
        public_id=str(uuid4()),
        system_prompt=system_prompt,
    )

    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)

    return chat_session


def get_session_by_public_id(
    db: Session,
    public_id: str,
) -> ChatSession:
    """Retrieve a session with its messages."""

    statement = (
        select(ChatSession)
        .options(selectinload(ChatSession.messages))
        .where(ChatSession.public_id == public_id)
    )

    chat_session = db.scalar(statement)

    if chat_session is None:
        raise SessionNotFoundError(public_id)

    return chat_session


def append_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
) -> ChatMessage:
    """Append a message with the next sequence number."""

    statement = (
        select(func.max(ChatMessage.sequence_number))
        .where(ChatMessage.session_id == session_id)
    )

    current_max = db.scalar(statement)

    next_sequence = (current_max or 0) + 1

    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sequence_number=next_sequence,
    )

    db.add(message)
    db.flush()

    return message


def update_session_tokens(
    db: Session,
    session_id: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Increment a session's token totals."""

    statement = (
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(
            total_input_tokens=ChatSession.total_input_tokens + input_tokens,
            total_output_tokens=ChatSession.total_output_tokens + output_tokens,
        )
    )

    db.execute(statement)


def record_llm_call(
    db: Session,
    session_id: int,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int,
    finish_reason: str | None,
) -> LLMCall:

    prices = GROQ_PRICES.get(model_name)

    if prices is None:
        raise ValueError(
            f"No pricing configured for model: {model_name}"
        )

    cost = (
        prices["input"] * input_tokens
        + prices["output"] * output_tokens
    )

    call = LLMCall(
        session_id=session_id,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        duration_ms=duration_ms,
        finish_reason=finish_reason,
    )

    db.add(call)
    db.flush()

    return call