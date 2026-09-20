"""Step 6.3 guard rails: what the dashboard reads and what it draws.

The queries are tested on SQLite and, when one is running, on real Postgres.
The page itself is run headless with Streamlit's AppTest, which executes the
script and fails on any exception the browser would have shown.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.data import (
    WINDOWS,
    bucket_seconds_for,
    busiest_cards,
    card_history,
    decisions_over_time,
    escalated,
    is_truncated,
    load_decisions,
    reason_counts,
    reasons_of,
    review_queue,
    risk_histogram,
    since_for,
    summarise,
)
from src.api.schemas import Decision, RiskResult, VelocityFeatures
from src.config import PROJECT_ROOT, settings
from src.storage import db
from src.storage.db import create_db_engine, create_schema, create_session_factory, session_scope
from src.storage.models import Base
from src.storage.repository import save_decisions

# Relative paths resolve against this file, so the page is named from the repo root.
APP = str(PROJECT_ROOT / "dashboard" / "app.py")
NOON = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)

VELOCITY = VelocityFeatures(
    count_1m=7, count_5m=9, count_1h=12, amount_1h=250.5, countries_1h=2, seconds_since_previous=1.0
)


def result(transaction_id, decision=Decision.APPROVE, risk=0.01, **overrides):
    fields = {
        "transaction_id": transaction_id,
        "risk_score": risk,
        "decision": decision,
        "review_threshold": 0.24,
        "block_threshold": 0.95,
        "model_version": "v-test",
        "card_id": "card-00001",
    }
    return RiskResult(**(fields | overrides))


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def engine(request, tmp_path):
    """An empty database: a SQLite file, or a throwaway schema in Postgres."""
    if request.param == "sqlite":
        sqlite = create_db_engine(f"sqlite:///{(tmp_path / 'dashboard.db').as_posix()}")
        yield sqlite
        sqlite.dispose()
    else:
        postgres = request.getfixturevalue("postgres_engine")
        yield postgres
        Base.metadata.drop_all(postgres)


@pytest.fixture
def populated(engine):
    """Three minutes of decisions: approvals, one review, one block, one escalation."""
    create_schema(engine)
    sessions = create_session_factory(engine)
    batches = {
        NOON: [result(f"tx-{i}") for i in range(5)],
        NOON
        + timedelta(minutes=1): [
            result("tx-review", Decision.REVIEW, risk=0.5),
            result(
                "tx-escalated",
                Decision.REVIEW,
                model_decision=Decision.APPROVE,
                reasons=["many_in_a_minute"],
                velocity=VELOCITY,
            ),
        ],
        NOON
        + timedelta(minutes=2): [
            result("tx-block", Decision.BLOCK, risk=0.99, card_id="card-00002")
        ],
    }
    for scored_at, results in batches.items():
        with session_scope(sessions) as session:
            save_decisions(session, results, scored_at=scored_at)
    return engine


# ------------------------------------------------------------------- reading
def test_every_decision_in_the_window_is_read(populated):
    frame = load_decisions(populated)
    assert len(frame) == 8
    assert set(frame["decision"]) == {"approve", "review", "block"}
    # UTC on both databases: SQLite drops the offset, and charts cannot mix the two.
    assert frame["scored_at"].dt.tz is not None


def test_a_window_only_reads_what_falls_inside_it(populated):
    frame = load_decisions(populated, since=NOON + timedelta(minutes=2))
    assert list(frame["transaction_id"]) == ["tx-block"]


def test_the_newest_decisions_come_first(populated):
    frame = load_decisions(populated)
    assert frame["scored_at"].is_monotonic_decreasing


def test_a_window_is_a_time_not_a_row_count():
    now = datetime(2026, 9, 16, 12, 30, tzinfo=timezone.utc)
    assert since_for(900, now) == now - timedelta(minutes=15)
    assert since_for(None, now) is None  # "Everything"


def test_the_page_says_when_it_is_showing_a_truncated_window(populated):
    """A chart of part of the data, drawn as if it were all of it, is a lie."""
    assert is_truncated(load_decisions(populated, limit=3), limit=3)
    assert not is_truncated(load_decisions(populated, limit=100), limit=100)


# --------------------------------------------------------------- summarising
def test_the_totals_add_up(populated):
    summary = summarise(load_decisions(populated))
    assert (summary.total, summary.approve, summary.review, summary.block) == (8, 5, 2, 1)
    assert summary.cards == 2
    assert summary.review_rate == pytest.approx(2 / 8)


def test_only_decisions_the_rules_changed_count_as_escalated(populated):
    """tx-review was the model's own call; tx-escalated was the rules'."""
    frame = load_decisions(populated)
    assert summarise(frame).escalated == 1
    assert list(frame.loc[escalated(frame), "transaction_id"]) == ["tx-escalated"]


def test_an_empty_window_summarises_to_zeroes(engine):
    create_schema(engine)
    summary = summarise(load_decisions(engine))
    assert (summary.total, summary.escalated, summary.per_second) == (0, 0, 0.0)
    assert summary.first is None


def test_throughput_is_measured_over_the_decisions_not_the_window(populated):
    """Two minutes of data in a 24-hour window is a two-minute rate."""
    summary = summarise(load_decisions(populated))
    assert summary.per_second == pytest.approx(8 / 120)


# ------------------------------------------------------------------ charting
def test_each_bucket_counts_its_own_decisions(populated):
    counted = decisions_over_time(load_decisions(populated), bucket_seconds=60)
    assert list(counted.index) == [NOON, NOON + timedelta(minutes=1), NOON + timedelta(minutes=2)]
    assert list(counted["approve"]) == [5, 0, 0]
    assert list(counted["review"]) == [0, 2, 0]
    assert list(counted["block"]) == [0, 0, 1]


def test_a_quiet_minute_is_a_zero_not_a_gap(populated):
    """Leaving empty buckets out would draw a line straight over an outage."""
    frame = load_decisions(populated)
    frame = frame[frame["transaction_id"] != "tx-review"]
    frame = frame[frame["transaction_id"] != "tx-escalated"]
    counted = decisions_over_time(frame, bucket_seconds=60)
    assert len(counted) == 3
    assert counted.iloc[1].sum() == 0


def test_bucket_width_comes_from_a_fixed_set(populated):
    """Bars are a second, a minute or an hour, never 7.3 seconds."""
    frame = load_decisions(populated)
    assert bucket_seconds_for(frame, target_buckets=60) == 5
    assert bucket_seconds_for(frame, target_buckets=2) == 60
    assert bucket_seconds_for(pd.DataFrame(columns=["scored_at"])) == 1


def test_an_empty_window_still_has_all_three_columns(engine):
    create_schema(engine)
    counted = decisions_over_time(load_decisions(engine), bucket_seconds=60)
    assert list(counted.columns) == ["approve", "review", "block"]


def test_risk_scores_are_binned_so_the_thresholds_are_visible(populated):
    """Almost every score is near zero; linear bins would hide everything above 0.02."""
    histogram = risk_histogram(load_decisions(populated), bins=20)
    assert histogram["count"].sum() == 8
    above_review = histogram.loc[histogram["risk_score"] >= 0.24, "count"].sum()
    assert above_review == 2  # the review at 0.5 and the block at 0.99


# -------------------------------------------------------- the queue (6.4)
def test_the_queue_holds_everything_a_human_must_look_at(populated):
    queue = review_queue(load_decisions(populated))
    assert set(queue["transaction_id"]) == {"tx-review", "tx-escalated", "tx-block"}
    assert queue["scored_at"].is_monotonic_decreasing  # newest first


def test_the_queue_can_show_only_what_the_rules_sent(populated):
    """The queue the model would never have produced on its own."""
    queue = review_queue(load_decisions(populated), only_escalated=True)
    assert list(queue["transaction_id"]) == ["tx-escalated"]


def test_each_queued_transaction_carries_its_rules(populated):
    queue = review_queue(load_decisions(populated))
    rules = dict(zip(queue["transaction_id"], queue["rules"], strict=True))
    assert rules["tx-escalated"] == ["many_in_a_minute"]
    assert rules["tx-review"] == []  # the model's own call, no rule fired


def test_a_decision_recorded_before_the_rules_existed_reads_as_no_rules(populated):
    """Phase 4 rows have a null reasons column; that must not crash the page."""
    frame = load_decisions(populated)
    frame.loc[frame["transaction_id"] == "tx-review", "reasons"] = None
    assert list(reasons_of(frame.head(1)))[0] == []


def test_rules_are_counted_over_the_whole_window(populated):
    counted = reason_counts(load_decisions(populated))
    assert list(counted["rule"]) == ["many_in_a_minute"]
    assert list(counted["count"]) == [1]


def test_an_empty_window_has_an_empty_queue(engine):
    create_schema(engine)
    frame = load_decisions(engine)
    assert review_queue(frame).empty
    assert reason_counts(frame).empty
    assert busiest_cards(frame).empty


# -------------------------------------------------------- busiest cards (6.4)
def test_cards_are_ranked_by_how_much_they_did(populated):
    cards = busiest_cards(load_decisions(populated))
    assert list(cards["card_id"]) == ["card-00001", "card-00002"]
    assert list(cards["transactions"]) == [7, 1]


def test_each_card_shows_what_the_service_did_about_it(populated):
    cards = busiest_cards(load_decisions(populated)).set_index("card_id")
    assert cards.loc["card-00001", "reviewed"] == 2
    assert cards.loc["card-00001", "escalated"] == 1
    assert cards.loc["card-00002", "blocked"] == 1
    assert cards.loc["card-00001", "peak_1m"] == 7  # from the velocity columns


def test_a_card_drills_down_to_its_own_transactions(populated):
    history = card_history(load_decisions(populated), "card-00002")
    assert list(history["transaction_id"]) == ["tx-block"]


def test_transactions_without_a_card_are_not_a_card(engine):
    """A transaction may arrive with no card at all; it belongs to no row here."""
    create_schema(engine)
    sessions = create_session_factory(engine)
    with session_scope(sessions) as session:
        save_decisions(session, [result("tx-anonymous", card_id=None)], scored_at=NOON)
    assert busiest_cards(load_decisions(engine)).empty


# ---------------------------------------------------------------- the page
@pytest.fixture
def app(populated, monkeypatch):
    """The real page, pointed at the test database instead of the one in .env.

    The engine is swapped rather than the URL: the Postgres one carries a
    throwaway search_path in its connect arguments, which a URL cannot.
    """
    monkeypatch.setattr(db, "create_db_engine", lambda *args, **kwargs: populated)
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=30)


def test_the_page_runs_without_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "Fraud decisions"


def test_the_page_shows_the_totals(app):
    """The seeded decisions are days old, so the page is asked for everything."""
    app.run()
    app.selectbox(key="window").set_value("Everything").run()
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Decisions"] == "8"
    assert labels["Reviewed"] == "2"
    assert labels["Escalated by velocity"] == "1"


def test_the_page_draws_its_charts(app):
    app.run()
    app.selectbox(key="window").set_value("Everything").run()
    # Decisions over time, risk scores, and which rules are filling the queue.
    assert len(app.get("plotly_chart")) == 3


def test_changing_the_window_re_reads(app):
    app.run()
    app.selectbox(key="window").set_value("Everything").run()
    assert app.metric[0].value == "8"
    # The seeded decisions are days old, so a short window has nothing to draw.
    app.selectbox(key="window").set_value("Last 15 minutes").run()
    assert not app.exception
    assert app.info[0].value.startswith("No decisions in this window")


def test_every_window_on_the_page_can_be_chosen(app):
    app.run()
    for window in WINDOWS:
        app.selectbox(key="window").set_value(window).run()
        assert not app.exception


def test_a_database_that_is_down_is_a_message_not_a_crash(monkeypatch):
    """The dashboard is a reader; it fails on its own, without taking anything with it."""
    monkeypatch.setattr(settings, "database_url", "postgresql+psycopg://x:x@127.0.0.1:1/none")
    st.cache_data.clear()
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert "Could not read the decision store" in app.error[0].value


def test_the_page_lists_the_queue_and_the_busiest_cards(app):
    app.run()
    app.selectbox(key="window").set_value("Everything").run()
    assert not app.exception
    # The queue, the card ranking, and the chosen card's own transactions.
    assert len(app.dataframe) == 3
    queued = app.dataframe[0].value
    assert set(queued["transaction_id"]) == {"tx-review", "tx-escalated", "tx-block"}


def test_the_page_can_show_only_escalated_transactions(app):
    app.run()
    app.selectbox(key="window").set_value("Everything").run()
    app.toggle(key="only_escalated").set_value(True).run()
    assert list(app.dataframe[0].value["transaction_id"]) == ["tx-escalated"]


def test_choosing_a_card_shows_that_card(app):
    app.run()
    app.selectbox(key="window").set_value("Everything").run()
    app.selectbox(key="card").set_value("card-00002").run()
    assert not app.exception
    assert list(app.dataframe[2].value["transaction_id"]) == ["tx-block"]
