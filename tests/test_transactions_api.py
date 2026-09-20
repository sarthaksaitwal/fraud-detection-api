"""Step 3.2 guard rails: recording decisions and reading them back over HTTP."""

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.features.redis_client import create_redis
from src.ml.preprocess import RAW_FEATURES
from src.scoring import Scorer
from src.storage.db import create_db_engine
from src.storage.store import DecisionStore


class AmountAsProbability:
    """Stands in for the model: fraud probability = Amount / 100, capped at 1."""

    def predict_proba(self, X):
        p = np.clip(X["Amount"].to_numpy() / 100, 0, 1)
        return np.column_stack([1 - p, p])


SCORER = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)


def serve(store):
    """The app with the fake model and the given store (None = recording off)."""
    return TestClient(create_app(load_scorer=lambda: SCORER, open_store=lambda: store))


def sqlite_store(path):
    return DecisionStore(create_db_engine(f"sqlite:///{path.as_posix()}"))


@pytest.fixture
def transaction(raw_df):
    row = raw_df[RAW_FEATURES].iloc[0].to_dict()

    def make(transaction_id, amount):
        return {**row, "transaction_id": transaction_id, "Amount": amount}

    return make


@pytest.fixture
def client(tmp_path):
    with serve(sqlite_store(tmp_path / "decisions.db")) as test_client:
        yield test_client


@pytest.fixture
def unreachable_client(tmp_path):
    """The app with a database that cannot be opened: its directory does not exist."""
    with serve(sqlite_store(tmp_path / "not-yet" / "decisions.db")) as test_client:
        yield test_client


def test_a_scored_transaction_can_be_looked_up(client, transaction):
    scored = client.post("/score", json=transaction("tx-1", 50)).json()

    response = client.get("/transactions/tx-1")
    assert response.status_code == 200
    stored = response.json()
    assert {key: stored[key] for key in scored} == scored
    assert stored["scored_at"]


def test_every_transaction_in_a_batch_is_recorded(client, transaction):
    batch = [transaction("tx-a", 10), transaction("tx-b", 50), transaction("tx-c", 95)]
    client.post("/score/batch", json={"transactions": batch})

    for tid, decision in [("tx-a", "approve"), ("tx-b", "review"), ("tx-c", "block")]:
        assert client.get(f"/transactions/{tid}").json()["decision"] == decision


def test_rescoring_a_transaction_replaces_its_decision(client, transaction):
    client.post("/score", json=transaction("tx-1", 10))
    client.post("/score", json=transaction("tx-1", 95))
    assert client.get("/transactions/tx-1").json()["decision"] == "block"
    assert len(client.get("/transactions").json()["decisions"]) == 1


def test_an_unknown_transaction_is_a_404(client):
    response = client.get("/transactions/never-seen")
    assert response.status_code == 404
    assert response.json() == {"detail": "no decision recorded for this transaction"}


def test_the_list_filters_by_decision_and_limit(client, transaction):
    batch = [transaction(f"tx-{i}", amount) for i, amount in enumerate([10, 95, 95, 95])]
    client.post("/score/batch", json={"transactions": batch})

    blocked = client.get("/transactions", params={"decision": "block", "limit": 2}).json()
    assert len(blocked["decisions"]) == 2
    assert {d["decision"] for d in blocked["decisions"]} == {"block"}


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 501}, {"limit": "ten"}, {"decision": "maybe"}]
)
def test_bad_list_parameters_are_a_422(client, params):
    assert client.get("/transactions", params=params).status_code == 422


def test_health_reports_a_working_store(client):
    assert client.get("/health").json()["decision_store"] == "ok"


def test_scoring_still_works_when_the_database_is_down(unreachable_client, transaction, caplog):
    body = {"transactions": [transaction("tx-2", 10)]}
    with caplog.at_level(logging.ERROR, logger="src.storage"):
        single = unreachable_client.post("/score", json=transaction("tx-1", 95))
        batch = unreachable_client.post("/score/batch", json=body)
    assert single.status_code == 200
    assert single.json()["decision"] == "block"
    assert batch.status_code == 200
    # The first request finds the database down; the second does not try again.
    assert caplog.text.count("decision store unavailable, retrying") == 1
    assert "1 decision(s) not recorded" in caplog.text


def test_reads_are_a_503_when_the_database_is_down(unreachable_client):
    for path in ("/transactions/tx-1", "/transactions"):
        response = unreachable_client.get(path)
        assert response.status_code == 503
        assert response.json() == {"detail": "decision store unavailable"}


def test_health_stays_ok_but_reports_the_store_unavailable(unreachable_client):
    response = unreachable_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["decision_store"] == "unavailable"


def test_reads_are_a_503_when_recording_is_switched_off(transaction):
    with serve(None) as client:
        assert client.post("/score", json=transaction("tx-1", 50)).status_code == 200
        response = client.get("/transactions/tx-1")
    assert response.status_code == 503
    assert response.json() == {"detail": "decisions are not being recorded"}


def test_the_store_is_closed_when_the_app_shuts_down(tmp_path):
    store = sqlite_store(tmp_path / "decisions.db")
    closed = []
    store.close = lambda: closed.append(True)
    with serve(store):
        assert not closed
    assert closed == [True]


def test_docs_describe_the_transaction_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/transactions", "/transactions/{transaction_id}"} <= paths.keys()


@pytest.mark.redis
def test_a_stored_decision_reads_back_with_its_velocity(tmp_path, transaction, redis_url):
    """Step 5.4: the row explains the decision, including what the card had just done."""
    store = sqlite_store(tmp_path / "decisions.db")
    app = create_app(
        load_scorer=lambda: SCORER,
        open_store=lambda: store,
        open_redis=lambda: create_redis(redis_url),
    )
    with TestClient(app) as client:
        body = {**transaction("tx-velocity", 10), "card_id": "card-read-1"}
        scored = client.post("/score", json=body).json()
        stored = client.get("/transactions/tx-velocity").json()
    store.close()
    assert stored["card_id"] == "card-read-1"
    assert stored["velocity"] == scored["velocity"]
    assert stored["velocity"]["count_1m"] == 1
