"""Reading and writing scored decisions (Step 3.1)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from src.api.schemas import Decision, RiskResult
from src.storage.models import DecisionRecord, utcnow

# Both databases speak INSERT ... ON CONFLICT DO UPDATE, but the construct that
# builds it is dialect-specific.
UPSERT = {"postgresql": postgresql.insert, "sqlite": sqlite.insert}

UPDATABLE = [c.name for c in DecisionRecord.__table__.columns if c.name != "transaction_id"]


def save_decisions(
    session: Session, results: Sequence[RiskResult], scored_at: datetime | None = None
) -> int:
    """Record each result, overwriting the row if the transaction was scored before.

    One statement for the whole list, so a 1,000-row batch is one round trip
    rather than 1,000. Every row in the call shares one `scored_at`: they were
    decided by one model call at one moment.

    The ids in `results` must be distinct -- Postgres refuses to update the same
    row twice in one statement. The API guarantees that; Phase 4's consumer will
    have to deduplicate a micro-batch before calling this.
    """
    if not results:
        return 0

    scored_at = scored_at or utcnow()
    rows = [
        {
            "transaction_id": result.transaction_id,
            "risk_score": result.risk_score,
            "decision": result.decision.value,
            "review_threshold": result.review_threshold,
            "block_threshold": result.block_threshold,
            "model_version": result.model_version,
            "scored_at": scored_at,
        }
        for result in results
    ]

    dialect = session.get_bind().dialect.name
    if dialect not in UPSERT:
        raise NotImplementedError(f"no upsert implemented for the {dialect!r} dialect")

    statement = UPSERT[dialect](DecisionRecord).values(rows)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["transaction_id"],
            set_={name: statement.excluded[name] for name in UPDATABLE},
        )
    )
    return len(rows)


def get_decision(session: Session, transaction_id: str) -> DecisionRecord | None:
    """The stored decision for one transaction, or None if it was never scored."""
    return session.get(DecisionRecord, transaction_id)


def list_decisions(
    session: Session, decision: Decision | None = None, limit: int = 50
) -> list[DecisionRecord]:
    """The most recent decisions, newest first, optionally of one kind only.

    Every row saved in one call shares a `scored_at`, so ties are guaranteed;
    the transaction id breaks them and keeps the order stable between calls.
    """
    query = select(DecisionRecord)
    if decision is not None:
        query = query.where(DecisionRecord.decision == decision.value)
    query = query.order_by(DecisionRecord.scored_at.desc(), DecisionRecord.transaction_id)
    return list(session.scalars(query.limit(limit)))


def count_decisions(session: Session) -> dict[Decision, int]:
    """How many stored decisions of each kind, including kinds with none."""
    query = select(DecisionRecord.decision, func.count()).group_by(DecisionRecord.decision)
    counts = dict(session.execute(query).tuples().all())
    return {decision: counts.get(decision.value, 0) for decision in Decision}
