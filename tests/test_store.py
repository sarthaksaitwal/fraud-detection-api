"""Steps 3.2 and 3.4 guard rails: the best-effort decision store."""

import logging

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import Decision, RiskResult
from src.config import settings
from src.storage import store as store_module
from src.storage.db import create_db_engine
from src.storage.store import DecisionStore, StoreUnavailableError, open_configured_store


def result(transaction_id="tx-1", decision=Decision.REVIEW):
    return RiskResult(
        transaction_id=transaction_id,
        risk_score=0.5,
        decision=decision,
        review_threshold=0.24,
        block_threshold=0.95,
        model_version="v-test",
    )


def sqlite_url(path):
    return f"sqlite:///{path.as_posix()}"


class FakeClock:
    """Time that only moves when a test says so."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock):
    decision_store = DecisionStore(
        create_db_engine(sqlite_url(tmp_path / "decisions.db")), retry_after=10, clock=clock
    )
    yield decision_store
    decision_store.close()


@pytest.fixture
def unreachable(tmp_path, clock):
    """A database that cannot be opened until its directory is created."""
    path = tmp_path / "not-yet" / "decisions.db"
    decision_store = DecisionStore(create_db_engine(sqlite_url(path)), retry_after=10, clock=clock)
    yield decision_store
    decision_store.close()


@pytest.fixture
def attempts(monkeypatch):
    """Counts how often the store actually goes to the database."""
    calls = []
    real = store_module.create_schema

    def counting(engine):
        calls.append(engine)
        real(engine)

    monkeypatch.setattr(store_module, "create_schema", counting)
    return calls


def test_recorded_decisions_can_be_read_back(store):
    assert store.record([result("tx-1"), result("tx-2", Decision.BLOCK)])
    assert store.get("tx-2").decision == "block"
    assert [d.transaction_id for d in store.recent(Decision.BLOCK)] == ["tx-2"]


def test_the_schema_is_created_on_first_use(store):
    assert store.get("tx-1") is None


def test_ping_reports_whether_the_database_answers(store, unreachable):
    assert store.ping()
    assert not unreachable.ping()


def test_recording_to_an_unreachable_database_logs_instead_of_raising(unreachable, caplog):
    with caplog.at_level(logging.ERROR, logger="src.storage"):
        assert unreachable.record([result()]) is False
    assert "unavailable, retrying in 10s (1 decision(s) not recorded" in caplog.text
    assert "OperationalError" in caplog.text


def test_reading_from_an_unreachable_database_raises(unreachable):
    with pytest.raises(SQLAlchemyError):
        unreachable.get("tx-1")


def test_after_a_failure_the_database_is_left_alone_until_the_window_passes(
    unreachable, clock, attempts
):
    unreachable.record([result("tx-1")])
    clock.now += 9
    assert unreachable.record([result("tx-2")]) is False
    assert not unreachable.ping()
    with pytest.raises(StoreUnavailableError):
        unreachable.get("tx-1")
    assert len(attempts) == 1

    clock.now += 1
    unreachable.record([result("tx-3")])
    assert len(attempts) == 2


def test_recording_recovers_and_reports_what_was_lost(tmp_path, unreachable, clock, caplog):
    unreachable.record([result("tx-1"), result("tx-2")])  # fails: 2 lost
    unreachable.record([result("tx-3")])  # skipped during the window: 1 more lost
    (tmp_path / "not-yet").mkdir()
    clock.now += 10

    with caplog.at_level(logging.WARNING, logger="src.storage"):
        assert unreachable.record([result("tx-4")]) is True
    assert "decision store is back; 3 decision(s) were not recorded" in caplog.text
    assert unreachable.get("tx-4") is not None


def test_a_successful_health_probe_also_ends_the_outage(tmp_path, unreachable, clock):
    unreachable.record([result()])
    (tmp_path / "not-yet").mkdir()
    clock.now += 10
    assert unreachable.ping()
    assert not unreachable.is_resting()


def test_failure_logs_contain_neither_sql_nor_row_values(unreachable, caplog):
    with caplog.at_level(logging.ERROR, logger="src.storage"):
        unreachable.record([result("secret-transaction-id")])
    assert "secret-transaction-id" not in caplog.text
    assert "INSERT" not in caplog.text


def test_no_store_when_persistence_is_switched_off():
    # conftest switches PERSIST_DECISIONS off for every test.
    assert open_configured_store() is None


def test_a_database_that_is_down_at_startup_is_only_a_warning(tmp_path, monkeypatch, caplog):
    url = sqlite_url(tmp_path / "not-yet" / "decisions.db")
    monkeypatch.setattr(settings, "persist_decisions", True)
    monkeypatch.setattr(store_module, "create_db_engine", lambda: create_db_engine(url))
    with caplog.at_level(logging.WARNING, logger="src.storage"):
        opened = open_configured_store()
    assert opened is not None
    assert "unavailable at startup" in caplog.text
    opened.close()
