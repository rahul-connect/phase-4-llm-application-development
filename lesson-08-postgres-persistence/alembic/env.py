"""Alembic migration environment for chat tables."""

from logging.config import fileConfig

from alembic import context

from sqlalchemy import engine_from_config, pool

from chat_api.config import get_settings
from chat_api.models import Base


# Alembic configuration object.
config = context.config


# Configure logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# Load the same database URL used by FastAPI.
settings = get_settings()

config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.replace("%", "%%"),
)


# SQLAlchemy metadata containing our models.
target_metadata = Base.metadata


def include_object(
    object,
    name,
    type_,
    reflected,
    compare_to,
):
    """Only include chat-related tables and their objects."""

    if type_ == "table":
        return name.startswith("chat_")

    if type_ in {
        "column",
        "index",
        "unique_constraint",
        "foreign_key_constraint",
    }:
        table = getattr(object, "table", None)

        if table is not None:
            return table.name.startswith("chat_")

    return True


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""

    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        include_object=include_object,
        version_table="chat_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a database connection."""

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {},
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        version_table="chat_alembic_version",
    )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()