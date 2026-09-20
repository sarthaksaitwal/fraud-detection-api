"""Steps 3.1 and 3.5 guard rails: the storage layer.

Every test runs twice: on SQLite, which needs nothing installed, and on real
Postgres, which is skipped unless one is running (`make up`). SQLite accepts
some things Postgres refuses, so the SQLite run alone proves less than it seems.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError, StatementError

from src.api.schemas import Decision, RiskResult, VelocityFeatures
from src.storage import repository
from src.storage.db import (
    add_missing_columns,
    create_db_engine,
    create_schema,
    create_session_factory,
    session_scope,
)
from src.storage.models import Base, DecisionRecord
from src.storage.repository import (
    count_decisions,
    get_decision,
    list_decisions,
    save_decisions,
)


def result(transaction_id="tx-1", risk=0.5, decision=Decision.REVIEW, **overrides):
    """A RiskResult exactly as the scoring core produces one."""
    fields = {
        "transaction_id": transaction_id,
        "risk_score": risk,
        "decision": decision,
        "review_threshold": 0.24,
        "block_threshold": 0.95,
        "model_version": "v-test",
    }
    return RiskResult(**(fields | overrides))


VELOCITY = VelocityFeatures(
    count_1m=3, count_5m=7, count_1h=12, amount_1h=250.5, countries_1h=2, seconds_since_previous=4.5
)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def engine(request, tmp_path):
    """An empty database: a SQLite file, or a throwaway schema in Postgres."""
    if request.param == "sqlite":
        sqlite = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
        yield sqlite
        sqlite.dispose()
    else:
        postgres = request.getfixturevalue("postgres_engine")
        yield postgres
        Base.metadata.drop_all(postgres)


@pytest.fixture
def sessions(engine):
    create_schema(engine)
    return create_session_factory(engine)


def test_a_saved_decision_reads_back_unchanged(sessions):
    scored_at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    with session_scope(sessions) as session:
        assert save_decisions(session, [result("tx-1", 0.87)], scored_at) == 1

    with session_scope(sessions) as session:
        stored = get_decision(session, "tx-1")
    assert stored.transaction_id == "tx-1"
    assert stored.risk_score == pytest.approx(0.87)
    assert stored.decision == "review"
    assert (stored.review_threshold, stored.block_threshold) == (0.24, 0.95)
    assert stored.model_version == "v-test"
    assert stored.scored_at == scored_at


def test_timestamps_come_back_as_utc(sessions):
    with session_scope(sessions) as session:
        save_decisions(session, [result()])
    with session_scope(sessions) as session:
        assert get_decision(session, "tx-1").scored_at.tzinfo is not None


def test_a_naive_timestamp_is_refused(sessions):
    with (
        pytest.raises((StatementError, ValueError), match="timezone-aware"),
        session_scope(sessions) as session,
    ):
        save_decisions(session, [result()], scored_at=datetime(2026, 9, 16, 12, 0))


def test_scoring_the_same_transaction_twice_updates_one_row(sessions):
    with session_scope(sessions) as session:
        save_decisions(session, [result("tx-1", 0.10, Decision.APPROVE)])
    with session_scope(sessions) as session:
        save_decisions(session, [result("tx-1", 0.99, Decision.BLOCK)])

    with session_scope(sessions) as session:
        assert len(list_decisions(session)) == 1
        stored = get_decision(session, "tx-1")
    assert stored.decision == "block"
    assert stored.risk_score == pytest.approx(0.99)


def test_an_unscored_transaction_has_no_decision(sessions):
    with session_scope(sessions) as session:
        assert get_decision(session, "never-seen") is None


def test_saving_nothing_writes_nothing(sessions):
    with session_scope(sessions) as session:
        assert save_decisions(session, []) == 0
        assert list_decisions(session) == []


def test_a_null_block_threshold_round_trips(sessions):
    with session_scope(sessions) as session:
        save_decisions(session, [result(block_threshold=None)])
    with session_scope(sessions) as session:
        assert get_decision(session, "tx-1").block_threshold is None


def test_list_returns_the_newest_decisions_first(sessions):
    base = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    with session_scope(sessions) as session:
        for minute in range(3):
            save_decisions(session, [result(f"tx-{minute}")], base + timedelta(minutes=minute))
    with session_scope(sessions) as session:
        assert [d.transaction_id for d in list_decisions(session)] == ["tx-2", "tx-1", "tx-0"]


def test_list_can_filter_by_decision(sessions):
    with session_scope(sessions) as session:
        save_decisions(
            session,
            [result("tx-a", 0.10, Decision.APPROVE), result("tx-b", 0.99, Decision.BLOCK)],
        )
    with session_scope(sessions) as session:
        blocked = list_decisions(session, decision=Decision.BLOCK)
    assert [d.transaction_id for d in blocked] == ["tx-b"]


def test_list_respects_the_limit(sessions):
    with session_scope(sessions) as session:
        save_decisions(session, [result(f"tx-{i}") for i in range(5)])
    with session_scope(sessions) as session:
        assert len(list_decisions(session, limit=2)) == 2


def test_the_database_refuses_a_decision_the_api_cannot_produce(sessions):
    with pytest.raises(IntegrityError), session_scope(sessions) as session:
        session.add(
            DecisionRecord(
                transaction_id="tx-1",
                risk_score=0.5,
                decision="maybe",
                review_threshold=0.24,
                block_threshold=0.95,
                model_version="v-test",
            )
        )


def test_a_failed_block_writes_nothing(sessions):
    with pytest.raises(RuntimeError), session_scope(sessions) as session:
        save_decisions(session, [result("tx-1")])
        raise RuntimeError("boom")

    with session_scope(sessions) as session:
        assert get_decision(session, "tx-1") is None


def test_every_risk_result_field_is_persisted(sessions):
    """Adding a field to the API response must not silently stop being recorded."""
    columns = set(DecisionRecord.__table__.columns.keys())
    # Velocity is the one field stored flattened: one column per feature (Step 5.4).
    assert set(RiskResult.model_fields) - {"velocity"} <= columns
    assert {f"velocity_{name}" for name in VelocityFeatures.model_fields} <= columns


def test_decisions_saved_together_share_one_timestamp(sessions, monkeypatch):
    # A clock that moves on every call, so a per-row timestamp cannot pass by luck.
    ticks = iter(range(1000))
    base = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(repository, "utcnow", lambda: base + timedelta(seconds=next(ticks)))

    with session_scope(sessions) as session:
        save_decisions(session, [result(f"tx-{i}") for i in range(5)])
    with session_scope(sessions) as session:
        assert {d.scored_at for d in list_decisions(session)} == {base}


def test_a_full_batch_is_written_in_one_statement(engine, sessions):
    statements = []

    def remember(connection, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", remember)
    try:
        with session_scope(sessions) as session:
            save_decisions(session, [result(f"tx-{i}") for i in range(1000)])
    finally:
        event.remove(engine, "before_cursor_execute", remember)
    assert sum(s.lstrip().upper().startswith("INSERT") for s in statements) == 1
    with session_scope(sessions) as session:
        assert len(list_decisions(session, limit=1000)) == 1000


def test_a_timestamp_in_another_timezone_is_stored_as_the_same_instant(sessions):
    india = timezone(timedelta(hours=5, minutes=30))
    scored_at = datetime(2026, 9, 16, 17, 30, tzinfo=india)
    with session_scope(sessions) as session:
        save_decisions(session, [result()], scored_at)
    with session_scope(sessions) as session:
        stored = get_decision(session, "tx-1").scored_at
    assert stored == datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    assert stored.utcoffset() == timedelta(0)


def test_the_database_refuses_a_risk_score_outside_0_to_1(sessions):
    with pytest.raises(IntegrityError), session_scope(sessions) as session:
        session.add(
            DecisionRecord(
                transaction_id="tx-1",
                risk_score=1.5,
                decision="block",
                review_threshold=0.24,
                block_threshold=0.95,
                model_version="v-test",
            )
        )


def test_postgres_refuses_the_same_id_twice_in_one_call(engine, sessions):
    """Why the Phase 4 consumer must deduplicate a micro-batch before saving it."""
    if engine.dialect.name != "postgresql":
        pytest.skip("SQLite quietly applies both rows; only Postgres refuses")
    with pytest.raises(StatementError), session_scope(sessions) as session:
        save_decisions(session, [result("tx-1"), result("tx-1", decision=Decision.BLOCK)])


def test_decisions_are_counted_by_kind(sessions):
    with session_scope(sessions) as session:
        assert count_decisions(session) == dict.fromkeys(Decision, 0)
        save_decisions(
            session,
            [result("tx-a", 0.99, Decision.BLOCK), result("tx-b", 0.99, Decision.BLOCK)]
            + [result("tx-c", 0.1, Decision.APPROVE)],
        )
    with session_scope(sessions) as session:
        counts = count_decisions(session)
    assert counts == {Decision.APPROVE: 1, Decision.REVIEW: 0, Decision.BLOCK: 2}


# ----------------------------------------------------------- velocity (Step 5.4)
def test_velocity_features_are_saved_with_the_decision(sessions):
    with session_scope(sessions) as session:
        save_decisions(session, [result(card_id="card-00001", velocity=VELOCITY)])
    with session_scope(sessions) as session:
        stored = get_decision(session, "tx-1")
        assert stored.card_id == "card-00001"
        assert stored.velocity == VELOCITY
        assert stored.velocity_count_1m == 3
        assert stored.velocity_seconds_since_previous == 4.5


def test_a_decision_made_without_velocity_stores_none(sessions):
    """Redis down, or a transaction with no card: the row says so rather than lying."""
    with session_scope(sessions) as session:
        save_decisions(session, [result()])
    with session_scope(sessions) as session:
        stored = get_decision(session, "tx-1")
        assert stored.card_id is None
        assert stored.velocity is None


def test_one_save_can_mix_decisions_with_and_without_velocity(sessions):
    """A single INSERT ... VALUES needs every row to have the same columns."""
    with session_scope(sessions) as session:
        save_decisions(
            session,
            [result("tx-1", card_id="card-00001", velocity=VELOCITY), result("tx-2")],
        )
    with session_scope(sessions) as session:
        assert get_decision(session, "tx-1").velocity == VELOCITY
        assert get_decision(session, "tx-2").velocity is None


def test_rescoring_replaces_the_velocity_of_the_earlier_decision(sessions):
    with session_scope(sessions) as session:
        save_decisions(session, [result(card_id="card-00001", velocity=VELOCITY)])
    later = VELOCITY.model_copy(update={"count_1m": 9})
    with session_scope(sessions) as session:
        save_decisions(session, [result(card_id="card-00001", velocity=later)])
    with session_scope(sessions) as session:
        assert get_decision(session, "tx-1").velocity.count_1m == 9


def test_columns_added_to_a_table_that_already_holds_rows(engine):
    """How Phase 5's columns reach a decisions table full of Phase 4 decisions."""
    velocity_columns = [c.name for c in DecisionRecord.__table__.columns if "velocity" in c.name]
    create_schema(engine)
    with engine.begin() as connection:
        for column in velocity_columns:
            connection.execute(text(f"ALTER TABLE decisions DROP COLUMN {column}"))

    # A decision recorded by the Phase 4 code, before the columns existed.
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO decisions (transaction_id, risk_score, decision, review_threshold,"
                " block_threshold, model_version, scored_at)"
                " VALUES ('older-decision', 0.5, 'review', 0.24, 0.95, 'v-test', :scored_at)"
            ),
            {"scored_at": datetime(2026, 9, 1, tzinfo=timezone.utc)},
        )

    assert sorted(add_missing_columns(engine)) == sorted(f"decisions.{c}" for c in velocity_columns)
    sessions = create_session_factory(engine)
    with session_scope(sessions) as session:
        # The old row survived and simply has no velocity.
        assert get_decision(session, "older-decision").velocity is None
        save_decisions(session, [result("newer-decision", velocity=VELOCITY)])
    with session_scope(sessions) as session:
        assert get_decision(session, "newer-decision").velocity == VELOCITY


def test_a_missing_column_that_cannot_be_null_is_refused(engine):
    """Adding a column with no value for existing rows is a real migration, not this."""
    create_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE decisions DROP COLUMN model_version"))
    with pytest.raises(RuntimeError, match="NOT NULL"):
        add_missing_columns(engine)


def test_the_rules_that_escalated_a_decision_are_stored(sessions):
    """Step 5.5: a review queue has to say why each transaction is in it."""
    escalated = result(
        decision=Decision.REVIEW,
        model_decision=Decision.APPROVE,
        reasons=["many_in_a_minute", "back_to_back"],
        card_id="card-00001",
        velocity=VELOCITY,
    )
    with session_scope(sessions) as session:
        save_decisions(session, [escalated])
    with session_scope(sessions) as session:
        stored = get_decision(session, "tx-1")
        assert stored.decision == "review"
        assert stored.model_decision == "approve"
        assert stored.reasons == ["many_in_a_minute", "back_to_back"]


def test_a_decision_the_rules_did_not_touch_stores_an_empty_list(sessions):
    with session_scope(sessions) as session:
        save_decisions(
            session, [result(decision=Decision.APPROVE, model_decision=Decision.APPROVE)]
        )
    with session_scope(sessions) as session:
        assert get_decision(session, "tx-1").reasons == []


def test_the_database_refuses_a_model_decision_that_is_not_one(sessions):
    with pytest.raises(IntegrityError), session_scope(sessions) as session:
        session.add(
            DecisionRecord(
                transaction_id="tx-1",
                risk_score=0.5,
                decision="review",
                review_threshold=0.24,
                block_threshold=0.95,
                model_version="v-test",
                scored_at=datetime(2026, 9, 16, tzinfo=timezone.utc),
                model_decision="maybe",
            )
        )
