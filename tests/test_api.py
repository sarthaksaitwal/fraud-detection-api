"""Step 2.3 guard rails: the HTTP API."""

import json
import logging
import math
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import MAX_BATCH_SIZE
from src.config import settings
from src.features.redis_client import create_redis
from src.ml.artifact import save_model
from src.ml.model import fit_xgboost, fraud_probability
from src.ml.preprocess import RAW_FEATURES, split
from src.scoring import Scorer

# Nothing listens on this port, so /health reports velocity unavailable at once.
UNREACHABLE_REDIS = "redis://127.0.0.1:6399/0"


class AmountAsProbability:
    """Stands in for the model: fraud probability = Amount / 100, capped at 1."""

    def predict_proba(self, X):
        p = np.clip(X["Amount"].to_numpy() / 100, 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def payload(raw_df):
    return raw_df[RAW_FEATURES].iloc[0].to_dict()


@pytest.fixture
def client():
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    with TestClient(create_app(load_scorer=lambda: scorer)) as test_client:
        yield test_client


def test_health_reports_the_model_being_served(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v-test",
        "review_threshold": 0.24,
        "block_threshold": 0.95,
        "decision_store": "disabled",
        "velocity": "disabled",
    }


def test_health_reports_velocity_unavailable_when_redis_is_down():
    """Step 5.1: the service is still healthy; it just has no recent-activity features."""
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(UNREACHABLE_REDIS))
    with TestClient(app) as test_client:
        body = test_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["velocity"] == "unavailable"


@pytest.mark.redis
def test_health_reports_velocity_ok_with_a_running_redis(redis_url):
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(redis_url))
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["velocity"] == "ok"


# ------------------------------------------------------- velocity (Step 5.4)
def test_a_score_has_no_velocity_when_the_feature_is_off(client, payload):
    body = client.post("/score", json={**payload, "card_id": "card-00001"}).json()
    assert body["card_id"] == "card-00001"
    assert body["velocity"] is None


@pytest.mark.redis
def test_a_score_reports_what_the_card_did_recently(redis_url, payload):
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(redis_url))
    with TestClient(app) as client:
        first = client.post("/score", json={**payload, "card_id": "card-api-1"}).json()
        second = client.post("/score", json={**payload, "card_id": "card-api-1"}).json()
    assert first["velocity"]["count_1m"] == 1
    assert first["velocity"]["seconds_since_previous"] is None
    # The card has been seen before now, and the gap to it is known.
    assert second["velocity"]["count_1m"] == 2
    assert second["velocity"]["seconds_since_previous"] >= 0


def test_scoring_carries_on_when_redis_is_down(payload, caplog):
    """Velocity is a signal, not a gate: losing it must not cost a payment."""
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(UNREACHABLE_REDIS))
    with TestClient(app) as client, caplog.at_level("WARNING"):
        response = client.post("/score", json={**payload, "Amount": 10, "card_id": "card-00001"})
    assert response.status_code == 200
    assert response.json()["decision"] == "approve"
    assert response.json()["velocity"] is None
    assert "velocity unavailable" in caplog.text


@pytest.mark.redis
def test_a_batch_measures_every_card_in_one_go(redis_url, batch, redis_client):
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(redis_url))
    batch = [{**row, "card_id": "card-api-3"} for row in batch]
    with TestClient(app) as client:
        results = client.post("/score/batch", json={"transactions": batch}).json()["results"]
    assert [r["velocity"]["count_1m"] for r in results] == [1, 2, 3]


@pytest.mark.redis
def test_a_card_going_too_fast_is_sent_for_review(redis_url, payload, monkeypatch):
    """Step 5.5: the same transaction, approved at first and reviewed once the card is busy."""
    monkeypatch.setattr(settings, "velocity_max_per_minute", 2)
    monkeypatch.setattr(settings, "velocity_min_gap_seconds", 0)  # only the count rule
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(redis_url))
    body = {**payload, "Amount": 10, "card_id": "card-fast"}
    with TestClient(app) as client:
        decisions = [
            client.post("/score", json={**body, "transaction_id": f"tx-{i}"}).json()
            for i in range(3)
        ]
    assert [d["decision"] for d in decisions] == ["approve", "approve", "review"]
    assert decisions[-1]["model_decision"] == "approve"
    assert decisions[-1]["reasons"] == ["many_in_a_minute"]
    assert decisions[-1]["risk_score"] == decisions[0]["risk_score"]  # the score is unchanged


@pytest.mark.parametrize(
    "amount, decision", [(10, "approve"), (24, "review"), (50, "review"), (95, "block")]
)
def test_score_returns_the_decision(client, payload, amount, decision):
    response = client.post("/score", json={**payload, "Amount": amount})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == decision
    assert body["risk_score"] == pytest.approx(amount / 100)
    assert body["model_version"] == "v-test"


def test_score_echoes_or_generates_the_transaction_id(client, payload):
    given = client.post("/score", json={**payload, "transaction_id": "tx-9"}).json()
    generated = client.post("/score", json=payload).json()
    assert given["transaction_id"] == "tx-9"
    assert generated["transaction_id"]


def test_missing_field_is_a_422_naming_the_field(client, payload):
    del payload["V14"]
    response = client.post("/score", json=payload)
    assert response.status_code == 422
    assert ["body", "V14"] in [error["loc"] for error in response.json()["detail"]]


def test_nan_in_the_request_body_is_a_422(client, payload):
    # json.dumps writes NaN, which is not valid JSON but which Python's parser accepts.
    body = json.dumps({**payload, "V3": math.nan})
    assert "NaN" in body
    response = client.post("/score", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "V3"]
    assert "input" not in error


def test_docs_describe_the_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/score", "/score/batch"} <= paths.keys()


def test_app_refuses_to_start_without_a_model(tmp_path):
    def missing_model():
        return Scorer.load(tmp_path / "missing.joblib", tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError), TestClient(create_app(load_scorer=missing_model)):
        pass


def test_real_model_served_over_http_matches_the_pipeline(raw_df, tmp_path):
    X_train, X_test, y_train, _ = split(raw_df)
    pipeline = fit_xgboost(X_train, y_train)
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    save_model(pipeline, {}, model_path, metadata_path)

    app = create_app(load_scorer=lambda: Scorer.load(model_path, metadata_path, 0.24, 0.95))
    with TestClient(app) as client:
        row = X_test.iloc[0].to_dict()
        body = client.post("/score", json=row).json()
    assert body["risk_score"] == pytest.approx(float(fraud_probability(pipeline, X_test)[0]))


@pytest.fixture
def batch(raw_df):
    """Three transactions the fake model approves, reviews and blocks."""
    rows = raw_df[RAW_FEATURES].head(3).to_dict(orient="records")
    return [
        {**row, "Amount": amount, "transaction_id": f"tx-{i}"}
        for i, (row, amount) in enumerate(zip(rows, [10, 50, 95], strict=True))
    ]


def test_batch_scores_every_transaction_in_order(client, batch):
    response = client.post("/score/batch", json={"transactions": batch})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["transaction_id"] for r in results] == ["tx-0", "tx-1", "tx-2"]
    assert [r["decision"] for r in results] == ["approve", "review", "block"]


def test_batch_and_single_endpoints_agree(client, batch):
    together = client.post("/score/batch", json={"transactions": batch}).json()["results"]
    one_by_one = [client.post("/score", json=row).json() for row in batch]
    assert together == one_by_one


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_outside_the_limits_is_a_422(client, payload, size):
    response = client.post("/score/batch", json={"transactions": [payload] * size})
    assert response.status_code == 422


def test_one_bad_transaction_rejects_the_whole_batch(client, batch):
    del batch[1]["V14"]
    response = client.post("/score/batch", json={"transactions": batch})
    assert response.status_code == 422
    locations = [error["loc"] for error in response.json()["detail"]]
    assert ["body", "transactions", 1, "V14"] in locations


def test_duplicate_transaction_ids_in_a_batch_are_a_422(client, batch):
    batch[2]["transaction_id"] = "tx-0"
    response = client.post("/score/batch", json={"transactions": batch})
    assert response.status_code == 422
    assert "tx-0" in response.json()["detail"][0]["msg"]


def test_unexpected_error_is_a_500_that_hides_the_details(payload):
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("secret internal detail")

    scorer = Scorer(BrokenModel(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/score", json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}


def test_every_response_carries_a_request_id(client):
    first, second = client.get("/health"), client.get("/health")
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_a_plain_incoming_request_id_is_kept(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("unsafe", ["has spaces", "a" * 65, "semi;colon"])
def test_an_unsafe_incoming_request_id_is_replaced(client, unsafe):
    response = client.get("/health", headers={"X-Request-ID": unsafe})
    assert response.headers["X-Request-ID"] != unsafe


def test_requests_are_logged_with_status_and_latency(client, payload, caplog):
    with caplog.at_level(logging.INFO, logger="src.api"):
        client.post("/score", json=payload, headers={"X-Request-ID": "req-1"})
        client.post("/score", json={}, headers={"X-Request-ID": "req-2"})
    assert re.search(r"POST /score 200 \d+\.\dms request_id=req-1", caplog.text)
    assert re.search(r"POST /score 422 \d+\.\dms request_id=req-2", caplog.text)


def test_request_logs_contain_no_transaction_values(client, payload, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post("/score", json={**payload, "Amount": 12.3456})
    assert "12.3456" not in caplog.text


def test_a_crashing_request_is_logged_as_500(payload, caplog):
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("secret internal detail")

    app = create_app(load_scorer=lambda: Scorer(BrokenModel(), "v-test", 0.24, 0.95))
    with (
        caplog.at_level(logging.INFO, logger="src.api"),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        client.post("/score", json=payload)
    assert re.search(r"POST /score 500 \d+\.\dms", caplog.text)
