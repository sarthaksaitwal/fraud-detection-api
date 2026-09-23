"""The fraud operations dashboard (Step 6.3).

    make dashboard                          # from the project root
    streamlit run dashboard/app.py          # the same thing, by hand
    cd dashboard && streamlit run app.py    # also works, see sys.path below

Reads the decisions the API and the consumer have recorded, straight from
Postgres. It is a reader: it opens its own pool, runs SELECTs, and is never in
the path of a decision, so a slow dashboard cannot slow a payment.

Queries live in dashboard/data.py, which is tested without a browser. This file
is layout and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts this file's own folder on sys.path, not the project root, so
# `dashboard.data` and `src` are only importable when the root happens to be
# there as well -- it is with `python -m streamlit`, and it is not with a bare
# `streamlit run app.py`. Adding it here makes the app run the same way from
# anywhere, before the first import that needs it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

from dashboard.data import (  # noqa: E402
    MAX_ROWS,
    WINDOWS,
    bucket_seconds_for,
    busiest_cards,
    card_history,
    decisions_over_time,
    escalated,
    is_truncated,
    load_decisions,
    reason_counts,
    review_queue,
    risk_histogram,
    since_for,
    summarise,
)
from src.config import settings  # noqa: E402
from src.features.rules import describe  # noqa: E402
from src.storage.db import create_db_engine  # noqa: E402
from src.storage.store import failure_reason  # noqa: E402

# The colours mean the same thing on every chart: green is money taken, amber is
# an analyst's time, red is a customer refused.
COLOURS = {"approve": "#2e9e5b", "review": "#e0a300", "block": "#d6453d"}
# How long a query's results are reused before the database is asked again. The
# page reruns on every widget click, and a dashboard is not worth a query storm.
CACHE_SECONDS = 10


@st.cache_resource
def engine():
    """One connection pool for the whole app, not one per rerun."""
    return create_db_engine()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Reading decisions…")
def decisions(window_seconds: int | None):
    """Decisions in the window, cached so a rerun does not re-query."""
    return load_decisions(engine(), since=since_for(window_seconds))


def decisions_chart(frame, bucket_seconds: int) -> go.Figure:
    counted = decisions_over_time(frame, bucket_seconds)
    figure = go.Figure()
    for decision, colour in COLOURS.items():
        figure.add_bar(x=counted.index, y=counted[decision], name=decision, marker_color=colour)
    figure.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.15),
        xaxis_title=None,
        yaxis_title=f"decisions per {humanise(bucket_seconds)}",
    )
    return figure


def risk_chart(frame) -> go.Figure:
    histogram = risk_histogram(frame)
    figure = go.Figure()
    figure.add_bar(x=histogram["risk_score"], y=histogram["count"], marker_color="#6c7a89")
    for threshold, label, colour in (
        (settings.review_threshold, "review", COLOURS["review"]),
        (settings.block_threshold, "block", COLOURS["block"]),
    ):
        if threshold is not None:
            figure.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=colour,
                annotation_text=f"{label} ≥ {threshold}",
                annotation_position="top",
            )
    figure.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        # Both axes are log: almost every transaction scores near zero, and on
        # linear axes the whole distribution collapses into the first bar.
        xaxis=dict(type="log", title="risk score"),
        yaxis=dict(type="log", title="transactions"),
    )
    return figure


def rules_chart(frame) -> go.Figure:
    counted = reason_counts(frame).sort_values("count")
    figure = go.Figure()
    figure.add_bar(
        x=counted["count"],
        y=counted["rule"],
        orientation="h",
        marker_color=COLOURS["review"],
        hovertext=[describe(rule) for rule in counted["rule"]],
        hoverinfo="text+x",
    )
    figure.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="transactions",
        yaxis_title=None,
    )
    return figure


QUEUE_COLUMNS = {
    "scored_at": st.column_config.DatetimeColumn("When", format="HH:mm:ss"),
    "transaction_id": st.column_config.TextColumn("Transaction"),
    "card_id": st.column_config.TextColumn("Card"),
    "decision": st.column_config.TextColumn("Decision"),
    "risk_score": st.column_config.NumberColumn("Risk", format="%.4f"),
    "velocity_count_1m": st.column_config.NumberColumn("In a minute"),
    "velocity_count_1h": st.column_config.NumberColumn("In an hour"),
    "velocity_countries_1h": st.column_config.NumberColumn("Countries"),
    "rules": st.column_config.ListColumn("Why it is here"),
}
CARD_COLUMNS = {
    "card_id": st.column_config.TextColumn("Card"),
    "transactions": st.column_config.NumberColumn("Transactions"),
    "reviewed": st.column_config.NumberColumn("Reviewed"),
    "blocked": st.column_config.NumberColumn("Blocked"),
    "escalated": st.column_config.NumberColumn("Escalated"),
    "peak_1m": st.column_config.NumberColumn("Peak in a minute"),
    "amount": st.column_config.NumberColumn("Amount in an hour", format="%.2f"),
}


def humanise(seconds: int) -> str:
    for size, name in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        if seconds >= size:
            count = seconds // size
            return name if count == 1 else f"{count} {name}s"
    return f"{seconds}s"


def main() -> None:
    st.set_page_config(page_title="Fraud decisions", page_icon="🛡️", layout="wide")
    st.title("Fraud decisions")

    with st.sidebar:
        st.caption("Reading " + engine().url.render_as_string(hide_password=True))
        window = st.selectbox("Window", list(WINDOWS), index=1, key="window")
        st.button("Refresh", width="stretch", on_click=st.cache_data.clear)
        st.caption(f"Cached for {CACHE_SECONDS}s; Refresh asks the database again.")

    try:
        frame = decisions(WINDOWS[window])
    except SQLAlchemyError as exc:
        st.error(f"Could not read the decision store: {failure_reason(exc)}")
        st.caption("Is it running? `docker compose up -d postgres`")
        return

    if is_truncated(frame):
        st.warning(
            f"Showing the newest {MAX_ROWS:,} decisions of a larger window. "
            "Pick a shorter window to see all of it."
        )
        frame = frame.head(MAX_ROWS)

    if frame.empty:
        st.info("No decisions in this window. Send some: `make stream`.")
        return

    summary = summarise(frame)
    columns = st.columns(5)
    columns[0].metric("Decisions", f"{summary.total:,}")
    columns[1].metric("Reviewed", f"{summary.review:,}", f"{summary.review_rate:.2%} of traffic")
    columns[2].metric("Blocked", f"{summary.block:,}", f"{summary.block_rate:.2%} of traffic")
    columns[3].metric(
        "Escalated by velocity",
        f"{summary.escalated:,}",
        f"{summary.escalated_rate:.2%} of traffic",
        help="Approvals the velocity rules sent for review instead.",
    )
    columns[4].metric(
        "Throughput",
        f"{summary.per_second:,.0f}/s",
        f"{summary.cards:,} cards",
        help="Over the span of these decisions, not the window.",
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Decisions over time")
        bucket = bucket_seconds_for(frame)
        st.plotly_chart(decisions_chart(frame, bucket), width="stretch")
    with right:
        st.subheader("Risk scores")
        st.plotly_chart(risk_chart(frame), width="stretch")
        st.caption(
            "The model's fraud probability for every transaction, against the thresholds "
            "that turn it into a decision."
        )

    velocity_share = escalated(frame).sum()
    st.caption(
        f"{summary.first:%Y-%m-%d %H:%M:%S} → {summary.last:%H:%M:%S} UTC · "
        f"{summary.approve:,} approve · {summary.review:,} review "
        f"({velocity_share:,} of them from velocity rules) · {summary.block:,} block"
    )

    queue_section(frame)
    cards_section(frame)


def queue_section(frame) -> None:
    """Everything the service did not approve, and why (Step 6.4)."""
    st.subheader("Review queue")
    only_escalated = st.toggle(
        "Only transactions the velocity rules sent",
        help="The queue the model would never have produced on its own.",
        key="only_escalated",
    )
    queue = review_queue(frame, only_escalated=only_escalated)
    if queue.empty:
        st.success("Nothing waiting for a human in this window.")
        return

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            queue.head(200)[list(QUEUE_COLUMNS)],
            column_config=QUEUE_COLUMNS,
            hide_index=True,
            height=320,
        )
        st.caption(
            f"{len(queue):,} waiting; showing the newest {min(len(queue), 200)}. "
            "Newest first, because a card being tested right now matters most."
        )
    with right:
        st.caption("Which rule put them there")
        st.plotly_chart(rules_chart(frame), width="stretch")
        for rule in reason_counts(frame)["rule"]:
            st.caption(f"**{rule}** — {describe(rule)}")


def cards_section(frame) -> None:
    """The cards doing the most, and what happened to one of them (Step 6.4)."""
    st.subheader("Busiest cards")
    cards = busiest_cards(frame, limit=10)
    if cards.empty:
        st.info("No cards in this window: these transactions carried no card id.")
        return

    st.dataframe(cards, column_config=CARD_COLUMNS, hide_index=True, height=240)
    chosen = st.selectbox("Look at a card", cards["card_id"], index=0, key="card")
    history = card_history(frame, chosen)
    st.dataframe(
        history.head(50)[list(QUEUE_COLUMNS)],
        column_config=QUEUE_COLUMNS,
        hide_index=True,
        height=240,
    )
    st.caption(
        f"{len(history):,} transaction(s) on {chosen} in this window; showing the newest "
        f"{min(len(history), 50)}."
    )


main()
