from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from decimal import Decimal

from sqlalchemy import Numeric


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    public_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
    )

    system_prompt: Mapped[str] = mapped_column(Text)

    total_input_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
    )

    total_output_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.sequence_number",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "sequence_number",
            name="uq_chat_messages_session_sequence",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    session_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "chat_sessions.id",
            ondelete="CASCADE",
        ),
        index=True,
    )

    role: Mapped[str] = mapped_column(String(20))

    content: Mapped[str] = mapped_column(Text)

    sequence_number: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    session: Mapped["ChatSession"] = relationship(
        back_populates="messages",
    )


class LLMCall(Base):
    __tablename__ = "chat_llm_calls"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    session_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "chat_sessions.id",
            ondelete="CASCADE",
        ),
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
    )

    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
    )

    duration_ms: Mapped[int] = mapped_column(
        Integer,
    )

    finish_reason: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )