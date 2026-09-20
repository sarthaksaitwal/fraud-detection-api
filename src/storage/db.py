"""Database engine, sessions and schema creation (Steps 3.1, 5.4)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.storage.models import Base

log = logging.getLogger("src.storage")

# Scoring waits on the database, so a database that stops answering must fail
# fast rather than hold every request for the operating system's TCP timeout.
POSTGRES_OPTIONS: dict[str, Any] = {
    "pool_size": 5,
    "max_overflow": 5,
    "pool_timeout": 5,  # seconds to wait for a free connection
    "pool_recycle": 1800,
    "connect_args": {"connect_timeout": 3},  # seconds to open a new connection
}


def create_db_engine(url: str | None = None, **options: Any) -> Engine:
    """The connection pool. Defaults to DATABASE_URL; tests pass a SQLite URL."""
    url = url or settings.database_url
    defaults: dict[str, Any] = {
        # Discard a connection the database dropped (a restart, an idle timeout)
        # instead of failing the request that happens to pick it up.
        "pool_pre_ping": True,
        # Keep row values out of error messages, which end up in the logs.
        "hide_parameters": True,
    }
    if not url.startswith("sqlite"):
        # SQLite needs no pool tuning and rejects these arguments.
        defaults |= POSTGRES_OPTIONS
    return create_engine(url, **(defaults | options))


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions that stay readable after they commit.

    expire_on_commit=False: by default SQLAlchemy marks every attribute stale
    on commit and re-queries on the next access, which fails once the session
    is closed. Rows we just wrote or read stay usable outside the block.
    """
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """A session that commits on success and always rolls back on failure."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_schema(engine: Engine) -> None:
    """Create missing tables, then add missing columns. Safe to repeat.

    Phase 5 adds columns to a `decisions` table that already holds rows, and
    create_all() only ever creates whole tables. Rather than ask everyone to
    drop their data, this adds any column the model has and the table does not.

    It is deliberately the smallest possible migration: adding a nullable column
    cannot fail on existing rows and cannot lose anything. A column that changes
    type, loses its NULLs or gets dropped needs a migration tool (Alembic), and
    this will not attempt it.
    """
    Base.metadata.create_all(engine)
    add_missing_columns(engine)


def add_missing_columns(engine: Engine) -> list[str]:
    """Add nullable columns the model declares but the table lacks. Returns their names."""
    inspector = inspect(engine)
    added = []
    for table in Base.metadata.sorted_tables:
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            if not column.nullable:
                raise RuntimeError(
                    f"{table.name}.{column.name} is missing and is NOT NULL; "
                    f"adding it needs a migration that says what existing rows should hold"
                )
            type_sql = column.type.compile(engine.dialect)
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {type_sql}")
                )
            added.append(f"{table.name}.{column.name}")
    if added:
        log.info("added missing column(s): %s", ", ".join(added))
    return added
