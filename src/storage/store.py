"""The decision store: what the API, and from Phase 4 the consumer, records to (Step 3.2).

Recording is best effort. If the database is down, `record` logs the failure
and returns False, and the caller still gets its score: refusing a customer's
payment because the audit log is unreachable is worse than a gap in the log.
Reads have nothing to fall back on, so they raise, and the API turns that into
a 503.

The store does not need the database to be up when the service starts. It
creates the schema on first use and retries on every call until that works,
so the service recovers by itself when the database comes back.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import Decision, RiskResult
from src.config import settings
from src.storage.db import create_db_engine, create_schema, create_session_factory, session_scope
from src.storage.models import DecisionRecord
from src.storage.repository import get_decision, list_decisions, save_decisions

log = logging.getLogger("src.storage")


def failure_reason(exc: SQLAlchemyError) -> str:
    """The driver's own message ("connection refused"), not the SQL statement.

    A failed 1,000-row insert would otherwise put the whole statement in the log.
    """
    return f"{type(exc).__name__}: {getattr(exc, 'orig', None) or exc}".strip()


class DecisionStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._sessions = create_session_factory(engine)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def ensure_schema(self) -> None:
        """Create the tables once. Raises SQLAlchemyError if the database is unreachable."""
        if self._schema_ready:
            return
        # Requests run in parallel threads; only one of them should run CREATE TABLE.
        with self._schema_lock:
            if not self._schema_ready:
                create_schema(self.engine)
                self._schema_ready = True

    def record(self, results: Sequence[RiskResult]) -> bool:
        """Save decisions. Never raises for a database failure; returns whether it worked."""
        try:
            self.ensure_schema()
            with session_scope(self._sessions) as session:
                save_decisions(session, results)
        except SQLAlchemyError as exc:
            log.error("could not record %d decision(s): %s", len(results), failure_reason(exc))
            return False
        return True

    def get(self, transaction_id: str) -> DecisionRecord | None:
        self.ensure_schema()
        with session_scope(self._sessions) as session:
            return get_decision(session, transaction_id)

    def recent(self, decision: Decision | None = None, limit: int = 50) -> list[DecisionRecord]:
        self.ensure_schema()
        with session_scope(self._sessions) as session:
            return list_decisions(session, decision, limit)

    def ping(self) -> bool:
        """Can the database be reached right now?"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return False
        return True

    def close(self) -> None:
        self.engine.dispose()


def open_configured_store() -> DecisionStore | None:
    """The store at DATABASE_URL, or None when PERSIST_DECISIONS is false.

    A wrong URL or a missing driver is a configuration mistake and raises, so the
    service refuses to start. A database that is merely down is only a warning.
    """
    if not settings.persist_decisions:
        log.info("not recording decisions: PERSIST_DECISIONS is false")
        return None
    store = DecisionStore(create_db_engine())
    try:
        store.ensure_schema()
    except SQLAlchemyError as exc:
        log.warning(
            "decision store unavailable at startup; scoring continues and recording "
            "retries on every request: %s",
            failure_reason(exc),
        )
    return store
