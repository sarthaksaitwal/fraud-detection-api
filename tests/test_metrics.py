"""Step 6.1 guard rails: what the service exports for Prometheus."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from prometheus_client.parser import text_string_to_metric_families

from src.api.main import create_app
from src.api.schemas import Decision, RiskResult
from src.features.redis_client import create_redis
from src.ml.preprocess import RAW_FEATURES
from src.observability.metrics import record_results
from src.scoring import Scorer


class AmountAsProbability:
    """Stands in for the model: fraud probability = Amount / 100, capped at 1."""

    def predict_proba(self, X):
        p = np.clip(X["Amount"].to_numpy() / 100, 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client():
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    with TestClient(create_app(load_scorer=lambda: scorer)) as test_client:
        yield test_client


@pytest.fixture
def payload(raw_df):
    return raw_df[RAW_FEATURES].iloc[0].to_dict()


def sample(text, name, **labels):
    """One metric value from the scraped text, or None if it is not there."""
    for family in text_string_to_metric_families(text):
        for metric in family.samples:
            if metric.name == name and labels.items() <= metric.labels.items():
                return metric.value
    return None


def scrape(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    return response.text


def result(transaction_id="tx-1", decision=Decision.APPROVE, **overrides):
    fields = {
        "transaction_id": transaction_id,
        "risk_score": 0.1,
        "decision": decision,
        "review_threshold": 0.24,
        "block_threshold": 0.95,
        "model_version": "v-test",
    }
    return RiskResult(**(fields | overrides))


# ------------------------------------------------------------------ the endpoint
def test_metrics_are_served_in_prometheus_format(client):
    assert "fraud_http_requests_total" in scrape(client)


def test_the_model_version_being_served_is_exported(client):
    assert sample(scrape(client), "fraud_model_loaded", model_version="v-test") == 1


def test_metrics_are_not_in_the_public_api_docs(client):
    """It is for a scraper, and its body is not JSON."""
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


# --------------------------------------------------------------------- requests
def test_requests_are_counted_by_route_and_status(client, payload):
    before = sample(scrape(client), "fraud_http_requests_total", route="/score", status="200") or 0
    client.post("/score", json=payload)
    after = sample(scrape(client), "fraud_http_requests_total", route="/score", status="200")
    assert after == before + 1


def test_a_rejected_request_is_counted_too(client, payload):
    del payload["V14"]
    client.post("/score", json=payload)
    assert sample(scrape(client), "fraud_http_requests_total", route="/score", status="422")


def test_latency_is_measured_per_route(client, payload):
    client.post("/score", json=payload)
    text = scrape(client)
    assert sample(text, "fraud_http_request_duration_seconds_count", route="/score") >= 1
    assert sample(text, "fraud_http_request_duration_seconds_sum", route="/score") > 0


def test_an_id_in_the_path_never_becomes_its_own_metric(client):
    """The label is the route template: one series, not one per transaction ever scored."""
    for transaction_id in ("tx-1", "tx-2", "tx-3"):
        client.get(f"/transactions/{transaction_id}")
    routes = {
        metric.labels["route"]
        for family in text_string_to_metric_families(scrape(client))
        for metric in family.samples
        if metric.name == "fraud_http_requests_total"
    }
    assert "/transactions/{transaction_id}" in routes
    assert not any(route.endswith(("tx-1", "tx-2", "tx-3")) for route in routes)


def test_a_url_that_matches_no_route_shares_one_series(client):
    client.get("/../../etc/passwd")
    client.get("/wp-admin.php")
    assert sample(scrape(client), "fraud_http_requests_total", route="unmatched", status="404")


def test_scrapes_do_not_measure_themselves(client):
    """A scrape runs on a timer; counting it would drown the rates it reports."""
    scrape(client)
    scrape(client)
    assert sample(scrape(client), "fraud_http_requests_total", route="/metrics") is None


# -------------------------------------------------------------------- decisions
def test_decisions_are_counted_by_kind_and_source():
    before = REGISTRY.get_sample_value(
        "fraud_decisions_total", {"decision": "block", "source": "api"}
    )
    record_results([result(decision=Decision.BLOCK, risk_score=0.99)], source="api")
    after = REGISTRY.get_sample_value(
        "fraud_decisions_total", {"decision": "block", "source": "api"}
    )
    assert after == (before or 0) + 1


def test_the_rules_that_escalated_a_decision_are_counted():
    before = REGISTRY.get_sample_value(
        "fraud_velocity_escalations_total", {"rule": "many_in_a_minute"}
    )
    record_results(
        [
            result(
                decision=Decision.REVIEW,
                model_decision=Decision.APPROVE,
                reasons=["many_in_a_minute"],
            )
        ],
        source="api",
    )
    after = REGISTRY.get_sample_value(
        "fraud_velocity_escalations_total", {"rule": "many_in_a_minute"}
    )
    assert after == (before or 0) + 1


def test_a_decision_the_rules_did_not_change_is_not_an_escalation():
    """The model reviewed it on its own score; the rules only agreed."""
    before = REGISTRY.get_sample_value("fraud_velocity_escalations_total", {"rule": "back_to_back"})
    record_results(
        [
            result(
                decision=Decision.REVIEW,
                model_decision=Decision.REVIEW,
                reasons=["back_to_back"],
            )
        ],
        source="api",
    )
    after = REGISTRY.get_sample_value("fraud_velocity_escalations_total", {"rule": "back_to_back"})
    assert after == before


def test_scoring_over_http_records_the_decision(client, payload):
    before = REGISTRY.get_sample_value(
        "fraud_decisions_total", {"decision": "approve", "source": "api"}
    )
    client.post("/score", json={**payload, "Amount": 1})
    after = REGISTRY.get_sample_value(
        "fraud_decisions_total", {"decision": "approve", "source": "api"}
    )
    assert after == (before or 0) + 1


def test_risk_scores_are_bucketed_at_the_thresholds(client, payload):
    """So "how much traffic is above the review threshold" reads straight off the histogram."""
    client.post("/score", json={**payload, "Amount": 1})
    text = scrape(client)
    assert sample(text, "fraud_risk_score_bucket", source="api", le="0.24") is not None
    assert sample(text, "fraud_risk_score_bucket", source="api", le="0.95") is not None


# --------------------------------------------------------------------- velocity
@pytest.mark.redis
def test_velocity_lookups_are_timed(redis_url, payload):
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(redis_url))
    before = REGISTRY.get_sample_value("fraud_velocity_duration_seconds_count") or 0
    with TestClient(app) as client:
        client.post("/score", json={**payload, "card_id": "card-metrics"})
    assert REGISTRY.get_sample_value("fraud_velocity_duration_seconds_count") == before + 1


def test_transactions_scored_without_velocity_are_counted(payload):
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    dead = "redis://127.0.0.1:6399/0"
    app = create_app(load_scorer=lambda: scorer, open_redis=lambda: create_redis(dead))
    before = REGISTRY.get_sample_value("fraud_velocity_skipped_total") or 0
    with TestClient(app) as client:
        client.post("/score", json={**payload, "card_id": "card-down"})
    assert REGISTRY.get_sample_value("fraud_velocity_skipped_total") == before + 1
