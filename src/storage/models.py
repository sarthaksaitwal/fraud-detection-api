"""The decisions table: one row per scored transaction (Step 3.1).

A decision is only explainable if its row carries the thresholds and the model
version that produced it. Thresholds live in .env and the model gets retrained,
so a row saying only "block" stops meaning anything the moment either changes.

The transaction's own features are deliberately not stored: they are the
sensitive part, and Phase 4's event stream is the right place for them. The
velocity columns Phase 5 adds are different in kind: they are counts of what the
card had already done, and without them a row cannot explain a decision the
velocity rules changed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.api.schemas import Decision, VelocityFeatures


def utcnow() -> datetime:
    """The current time, always timezone-aware UTC."""
    return datetime.now(timezone.utc)


class UtcDateTime(TypeDecorator):
    """A timestamp always stored and read back as timezone-aware UTC.

    SQLite has no timezone type: it drops the offset on write and hands back a
    naive datetime, while Postgres returns an aware one. Without this adapter
    the same code would produce different objects on each database, and the
    fast SQLite tests would stop being evidence about the real one.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware; use utcnow()")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base. `Base.metadata` is what creates the schema."""


class DecisionRecord(Base):
    """One scored transaction.

    The transaction id is the primary key, so re-scoring a transaction updates
    its row instead of adding a second one. That matters from Phase 4 onward:
    Kafka delivers at least once, so the consumer will occasionally replay a
    message it has already handled.
    """

    __tablename__ = "decisions"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    review_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    block_threshold: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=utcnow)

    # ------------------------------------------------- velocity (Steps 5.4, 5.5)
    # All nullable: a transaction can arrive without a card, and Redis can be
    # down, in which case the decision was made without these.
    card_id: Mapped[str | None] = mapped_column(String(64))
    velocity_count_1m: Mapped[int | None] = mapped_column(Integer)
    velocity_count_5m: Mapped[int | None] = mapped_column(Integer)
    velocity_count_1h: Mapped[int | None] = mapped_column(Integer)
    velocity_amount_1h: Mapped[float | None] = mapped_column(Float)
    velocity_countries_1h: Mapped[int | None] = mapped_column(Integer)
    velocity_seconds_since_previous: Mapped[float | None] = mapped_column(Float)
    # What the model alone decided, and which velocity rules changed it (Step 5.5).
    # Keeping both makes "how much extra review did the rules ask for?" one query.
    model_decision: Mapped[str | None] = mapped_column(String(16))
    reasons: Mapped[list[str] | None] = mapped_column(JSON)

    __table_args__ = (
        # Generated from the enum, so the database and the API can never disagree
        # about which decisions exist.
        CheckConstraint(
            f"decision IN ({', '.join(repr(d.value) for d in Decision)})",
            name="ck_decisions_decision",
        ),
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="ck_decisions_risk_score"),
        CheckConstraint(
            f"model_decision IS NULL OR model_decision IN "
            f"({', '.join(repr(d.value) for d in Decision)})",
            name="ck_decisions_model_decision",
        ),
        # "recent decisions", and "recent decisions of one kind" for the dashboard.
        Index("ix_decisions_scored_at", "scored_at"),
        Index("ix_decisions_decision_scored_at", "decision", "scored_at"),
    )

    @property
    def velocity(self) -> VelocityFeatures | None:
        """The velocity columns as one object, or None if the decision had none.

        StoredDecision reads this by name, so the API answers with the same
        shape /score returned when the decision was made.
        """
        if self.velocity_count_1h is None:
            return None
        return VelocityFeatures(
            count_1m=self.velocity_count_1m,
            count_5m=self.velocity_count_5m,
            count_1h=self.velocity_count_1h,
            amount_1h=self.velocity_amount_1h,
            countries_1h=self.velocity_countries_1h,
            seconds_since_previous=self.velocity_seconds_since_previous,
        )
