"""What the dashboard reads, and how it aggregates it (Step 6.3).

Kept apart from the Streamlit page so the queries can be tested without a
browser, and so the page stays layout.

The dashboard is a **reader**. It opens its own connection pool, runs SELECTs,
and is never in the path of a decision: a slow or broken dashboard can no more
delay a payment than closing the browser tab can.

Aggregation happens in pandas rather than SQL. Time bucketing is the one thing
SQLite and Postgres spell differently, and doing it here keeps the same code
tested on both. The cost is that rows cross the wire, so a window is capped at
MAX_ROWS and the page says when it is showing a capped view.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import Engine, desc, select

from src.api.schemas import Decision
from src.storage.models import DecisionRecord

# Windows offered on the page, newest first. None means "everything recorded".
WINDOWS: dict[str, int | None] = {
    "Last 15 minutes": 15 * 60,
    "Last hour": 60 * 60,
    "Last 24 hours": 24 * 60 * 60,
    "Everything": None,
}

# Enough for a busy day of this demo; past it the page says it is truncated
# rather than quietly drawing a chart of part of the data.
MAX_ROWS = 200_000

COLUMNS = (
    DecisionRecord.transaction_id,
    DecisionRecord.scored_at,
    DecisionRecord.decision,
    DecisionRecord.model_decision,
    DecisionRecord.risk_score,
    DecisionRecord.card_id,
    DecisionRecord.reasons,
    DecisionRecord.velocity_count_1m,
    DecisionRecord.velocity_count_1h,
    DecisionRecord.velocity_countries_1h,
    DecisionRecord.velocity_amount_1h,
)


@dataclass(frozen=True)
class Summary:
    """The numbers on the top row of the page."""

    total: int
    approve: int
    review: int
    block: int
    escalated: int  # approvals the velocity rules sent for review
    cards: int
    per_second: float
    first: datetime | None
    last: datetime | None

    @property
    def review_rate(self) -> float:
        return self.review / self.total if self.total else 0.0

    @property
    def block_rate(self) -> float:
        return self.block / self.total if self.total else 0.0

    @property
    def escalated_rate(self) -> float:
        return self.escalated / self.total if self.total else 0.0


def since_for(window_seconds: int | None, now: datetime | None = None) -> datetime | None:
    """The oldest `scored_at` a window includes, or None for everything."""
    if window_seconds is None:
        return None
    return (now or datetime.now(timezone.utc)) - timedelta(seconds=window_seconds)


def load_decisions(
    engine: Engine, since: datetime | None = None, limit: int = MAX_ROWS
) -> pd.DataFrame:
    """Recorded decisions, newest first, one row each.

    `limit + 1` rows are fetched so the caller can tell a full window from a
    truncated one without counting the table twice.
    """
    query = select(*COLUMNS).order_by(desc(DecisionRecord.scored_at)).limit(limit + 1)
    if since is not None:
        query = query.where(DecisionRecord.scored_at >= since)
    with engine.connect() as connection:
        frame = pd.read_sql(query, connection)
    # SQLite hands back naive datetimes; Postgres aware ones. Charts and window
    # arithmetic need one answer, and UTC is what the column holds.
    frame["scored_at"] = pd.to_datetime(frame["scored_at"], utc=True)
    return frame


def is_truncated(frame: pd.DataFrame, limit: int = MAX_ROWS) -> bool:
    return len(frame) > limit


def summarise(frame: pd.DataFrame) -> Summary:
    """Totals for a window. Rate is over the span of the rows, not the window."""
    counts = frame["decision"].value_counts() if not frame.empty else pd.Series(dtype=int)
    first = frame["scored_at"].min() if not frame.empty else None
    last = frame["scored_at"].max() if not frame.empty else None
    seconds = (last - first).total_seconds() if first is not None and last is not None else 0.0
    return Summary(
        total=len(frame),
        approve=int(counts.get(Decision.APPROVE.value, 0)),
        review=int(counts.get(Decision.REVIEW.value, 0)),
        block=int(counts.get(Decision.BLOCK.value, 0)),
        escalated=int(escalated(frame).sum()) if not frame.empty else 0,
        cards=int(frame["card_id"].nunique()) if not frame.empty else 0,
        per_second=len(frame) / seconds if seconds > 0 else 0.0,
        first=first,
        last=last,
    )


def escalated(frame: pd.DataFrame) -> pd.Series:
    """Which rows the velocity rules changed, rather than the model deciding alone."""
    if frame.empty:
        return pd.Series(dtype=bool)
    return frame["model_decision"].notna() & (frame["model_decision"] != frame["decision"])


def bucket_seconds_for(frame: pd.DataFrame, target_buckets: int = 60) -> int:
    """A bucket width that gives roughly `target_buckets` bars, from a fixed set.

    A fixed set keeps the axis readable: bars are a second, a minute, an hour --
    never 7.3 seconds.
    """
    choices = (1, 5, 15, 30, 60, 300, 900, 3600, 21600, 86400)
    if frame.empty:
        return choices[0]
    span = (frame["scored_at"].max() - frame["scored_at"].min()).total_seconds()
    for seconds in choices:
        if span / seconds <= target_buckets:
            return seconds
    return choices[-1]


def decisions_over_time(frame: pd.DataFrame, bucket_seconds: int) -> pd.DataFrame:
    """Counts per time bucket, one column per decision, oldest first.

    Empty buckets are filled with zeroes: a gap in a stacked chart means "no
    transactions", and leaving it out would draw a line straight over an outage.
    """
    kinds = [d.value for d in Decision]
    if frame.empty:
        return pd.DataFrame(columns=kinds, index=pd.DatetimeIndex([], name="scored_at"))
    counted = (
        frame.set_index("scored_at")
        .groupby([pd.Grouper(freq=f"{bucket_seconds}s"), "decision"])
        .size()
        .unstack("decision")
        .reindex(columns=kinds)
        .fillna(0)
        .astype(int)
    )
    return counted.asfreq(f"{bucket_seconds}s", fill_value=0).sort_index()


def risk_histogram(frame: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    """Risk scores in log-spaced bins.

    Almost every transaction scores near zero, so linear bins put 99% of the
    data in the first bar and tell you nothing about where the thresholds sit.
    """
    if frame.empty:
        return pd.DataFrame(columns=["risk_score", "count"])
    lowest = 1e-6
    edges = [0.0, *(10 ** (i / bins * 6 - 6) for i in range(1, bins + 1))]
    scores = frame["risk_score"].clip(lower=0, upper=1)
    counted = pd.cut(scores.where(scores > lowest, lowest), bins=edges, include_lowest=True)
    histogram = counted.value_counts().sort_index()
    return pd.DataFrame(
        {
            "risk_score": [interval.right for interval in histogram.index],
            "count": histogram.to_numpy(),
        }
    )


# ----------------------------------------------------- the queue (Step 6.4)
def reasons_of(frame: pd.DataFrame) -> pd.Series:
    """The velocity rules on each row, as a list.

    The column is JSON, and it is null for anything recorded before Phase 5, so
    a missing value has to read as "no rules", not as a crash.
    """
    if frame.empty:
        return pd.Series(dtype=object)
    return frame["reasons"].apply(lambda value: list(value) if isinstance(value, list) else [])


def review_queue(frame: pd.DataFrame, only_escalated: bool = False) -> pd.DataFrame:
    """What a human has to look at: everything the service did not approve.

    Newest first, because a card being tested right now matters more than one
    from an hour ago. `only_escalated` keeps just the ones the velocity rules
    sent, which is the queue the model would never have produced on its own.
    """
    if frame.empty:
        return frame
    queued = frame[frame["decision"].isin([Decision.REVIEW.value, Decision.BLOCK.value])].copy()
    if only_escalated:
        queued = queued[escalated(queued)]
    queued["rules"] = reasons_of(queued)
    return queued.sort_values("scored_at", ascending=False)


def reason_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """How often each rule fired, most first. One row can fire several."""
    fired = [rule for rules in reasons_of(frame) for rule in rules]
    if not fired:
        return pd.DataFrame(columns=["rule", "count"])
    counted = pd.Series(fired).value_counts()
    return pd.DataFrame({"rule": counted.index, "count": counted.to_numpy()})


def busiest_cards(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """The cards doing the most, with what the service did about them.

    Ordered by transactions in the window, not by velocity counts: a card at the
    top of this table is the one an analyst should look at first.
    """
    columns = ["card_id", "transactions", "reviewed", "blocked", "escalated", "peak_1m", "amount"]
    if frame.empty or frame["card_id"].isna().all():
        return pd.DataFrame(columns=columns)
    working = frame.dropna(subset=["card_id"]).copy()
    working["escalated"] = escalated(working)
    grouped = working.groupby("card_id").agg(
        transactions=("transaction_id", "count"),
        reviewed=("decision", lambda values: (values == Decision.REVIEW.value).sum()),
        blocked=("decision", lambda values: (values == Decision.BLOCK.value).sum()),
        escalated=("escalated", "sum"),
        peak_1m=("velocity_count_1m", "max"),
        amount=("velocity_amount_1h", "max"),
    )
    ordered = grouped.sort_values(["transactions", "reviewed"], ascending=False).head(limit)
    return ordered.reset_index()[columns]


def card_history(frame: pd.DataFrame, card_id: str) -> pd.DataFrame:
    """One card's transactions in the window, newest first."""
    if frame.empty:
        return frame
    history = frame[frame["card_id"] == card_id].copy()
    history["rules"] = reasons_of(history)
    return history.sort_values("scored_at", ascending=False)
