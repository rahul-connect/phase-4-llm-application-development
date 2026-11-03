"""FastAPI entry point."""

from fastapi import FastAPI

from chat_api.config import get_settings
from chat_api.exceptions import register_exception_handlers
from chat_api.routes import router


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

register_exception_handlers(app)

app.include_router(router)