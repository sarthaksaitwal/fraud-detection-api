"""The decision store: what the API, and from Phase 4 the consumer, records to (Steps 3.2, 3.4).

Recording is best effort. If the database is down, `record` logs the failure
and returns False, and the caller still gets its score: refusing a customer's
payment because the audit log is unreachable is worse than a gap in the log.
Reads have nothing to fall back on, so they raise, and the API turns that into
a 503.

A database that is down is slow to fail, not quick: resolving a host that has
gone away took about 4 seconds in Docker. So after a failure the store stops
trying for RETRY_AFTER_SECONDS and fails instantly instead. Only one request
per window pays for finding out whether the database is back.

The store does not need the database to be up when the service starts. It
creates the schema on first use, so the service recovers by itself.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.schemas import Decision, RiskResult
from src.config import settings
from src.storage.db import create_db_engine, create_schema, create_session_factory, session_scope
from src.storage.models import DecisionRecord
from src.storage.repository import get_decision, list_decisions, save_decisions

log = logging.getLogger("src.storage")

RETRY_AFTER_SECONDS = 10.0

T = TypeVar("T")


class StoreUnavailableError(Exception):
    """The database failed recently, so the store is not trying it again yet."""


def failure_reason(exc: Exception) -> str:
    """The driver's own message ("connection refused"), not the SQL statement.

    A failed 1,000-row insert would otherwise put the whole statement in the log.
    """
    return f"{type(exc).__name__}: {getattr(exc, 'orig', None) or exc}".strip()


class DecisionStore:
    def __init__(
        self,
        engine: Engine,
        retry_after: float = RETRY_AFTER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self._sessions = create_session_factory(engine)
        self._retry_after = retry_after
        self._clock = clock
        self._lock = threading.Lock()
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._down_until = 0.0  # clock time before which the database is not tried
        self._unrecorded = 0  # decisions lost since the database last worked

    # ------------------------------------------------------------- availability
    def is_resting(self) -> bool:
        """True while the store is waiting out the retry window after a failure."""
        return self._clock() < self._down_until

    def _failed(self, exc: SQLAlchemyError, lost: int = 0) -> None:
        with self._lock:
            self._down_until = self._clock() + self._retry_after
            self._unrecorded += lost
            unrecorded = self._unrecorded
        log.error(
            "decision store unavailable, retrying in %.0fs (%d decision(s) not recorded "
            "so far): %s",
            self._retry_after,
            unrecorded,
            failure_reason(exc),
        )

    def _succeeded(self) -> None:
        with self._lock:
            unrecorded, self._unrecorded = self._unrecorded, 0
        if unrecorded:
            log.warning("decision store is back; %d decision(s) were not recorded", unrecorded)

    def _use_database(self, work: Callable[[Session], T], lost_on_failure: int = 0) -> T:
        """Run `work` in a session, keeping track of whether the database is up."""
        if self.is_resting():
            with self._lock:
                self._unrecorded += lost_on_failure
            raise StoreUnavailableError("decision store unavailable")
        try:
            self.ensure_schema()
            with session_scope(self._sessions) as session:
                outcome = work(session)
        except SQLAlchemyError as exc:
            self._failed(exc, lost_on_failure)
            raise
        self._succeeded()
        return outcome

    # --------------------------------------------------------------- operations
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
            self._use_database(lambda s: save_decisions(s, results), lost_on_failure=len(results))
        except (SQLAlchemyError, StoreUnavailableError):
            return False
        return True

    def get(self, transaction_id: str) -> DecisionRecord | None:
        return self._use_database(lambda s: get_decision(s, transaction_id))

    def recent(self, decision: Decision | None = None, limit: int = 50) -> list[DecisionRecord]:
        return self._use_database(lambda s: list_decisions(s, decision, limit))

    def ping(self) -> bool:
        """Can the database be reached? Answers instantly during the retry window."""
        try:
            self._use_database(lambda s: s.execute(text("SELECT 1")))
        except (SQLAlchemyError, StoreUnavailableError):
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
            "retries on the next request: %s",
            failure_reason(exc),
        )
    return store
