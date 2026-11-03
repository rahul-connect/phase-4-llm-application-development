"""Custom exceptions and FastAPI exception handlers."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


class SessionNotFoundError(Exception):
    """Raised when a requested chat session does not exist."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

        super().__init__(
            f"Session {session_id} not found"
        )


async def session_not_found_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert SessionNotFoundError into a consistent 404 response."""

    assert isinstance(exc, SessionNotFoundError)

    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "session_not_found",
                "message": str(exc),
                "details": {
                    "session_id": exc.session_id,
                },
            }
        },
    )


async def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Convert HTTPException into the application's error shape."""

    assert isinstance(exc, HTTPException)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": str(exc.detail),
                "details": None,
            }
        },
    )


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """Register custom exception handlers on the FastAPI app."""

    app.add_exception_handler(
        SessionNotFoundError,
        session_not_found_handler,
    )

    app.add_exception_handler(
        HTTPException,
        http_exception_handler,
    )