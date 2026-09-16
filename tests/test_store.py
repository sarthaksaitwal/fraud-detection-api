"""Step 3.2 guard rails: the best-effort decision store."""

import logging

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import Decision, RiskResult
from src.config import settings
from src.storage import store as store_module
from src.storage.db import create_db_engine
from src.storage.store import DecisionStore, open_configured_store


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


@pytest.fixture
def store(tmp_path):
    decision_store = DecisionStore(create_db_engine(sqlite_url(tmp_path / "decisions.db")))
    yield decision_store
    decision_store.close()


@pytest.fixture
def unreachable(tmp_path):
    """A database that cannot be opened yet: its directory does not exist."""
    path = tmp_path / "not-yet" / "decisions.db"
    decision_store = DecisionStore(create_db_engine(sqlite_url(path)))
    yield decision_store
    decision_store.close()


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
    assert "could not record 1 decision(s): OperationalError" in caplog.text


def test_reading_from_an_unreachable_database_raises(unreachable):
    with pytest.raises(SQLAlchemyError):
        unreachable.get("tx-1")


def test_recording_recovers_when_the_database_comes_back(tmp_path, unreachable):
    assert unreachable.record([result("tx-1")]) is False
    (tmp_path / "not-yet").mkdir()
    assert unreachable.record([result("tx-2")]) is True
    assert unreachable.get("tx-2") is not None


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
