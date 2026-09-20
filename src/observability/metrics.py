"""The numbers a Prometheus server scrapes (Step 6.1).

Logs say what happened to one transaction; metrics say what is happening to all
of them. These are the ones worth waking someone for: how many requests, how
slow, what the service decided, and whether velocity is answering.

Two rules shape everything here.

**Labels must be bounded.** Every distinct combination of label values is a
separate time series kept in memory, so a label may never carry a transaction
id, a card id or a raw URL path. `/transactions/{transaction_id}` is one series;
`/transactions/test-row-5` would be one series per transaction ever scored.

**Measuring must not be able to break scoring.** Counters are incremented next
to the work, never inside it, and a metric is never read back to make a
decision.

One process, one set of numbers: these live in the default registry of whichever
process imports them, so the API and the consumer export their own. Running the
API under several uvicorn workers would need prometheus_client's multiprocess
mode, which this project does not use.
"""

from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import Counter, Gauge, Histogram

from src.api.schemas import RiskResult

# Buckets in seconds, chosen around what this service actually does: scoring is
# ~20 ms, a 1,000-row batch is ~400 ms. Prometheus' defaults spend most of their
# buckets above 1 s, where nothing of ours lives.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
# Redis is meant to answer in about a millisecond, so these are far tighter.
VELOCITY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)

# ------------------------------------------------------------------ requests
requests_total = Counter(
    "fraud_http_requests_total",
    "HTTP requests handled, by route template, method and status.",
    ["method", "route", "status"],
)
request_duration_seconds = Histogram(
    "fraud_http_request_duration_seconds",
    "Time to handle a request, measured in the server.",
    ["method", "route"],
    buckets=LATENCY_BUCKETS,
)

# ----------------------------------------------------------------- decisions
decisions_total = Counter(
    "fraud_decisions_total",
    "Decisions made, by what was decided and which way the transaction arrived.",
    ["decision", "source"],
)
escalations_total = Counter(
    "fraud_velocity_escalations_total",
    "Approvals the velocity rules sent for review, by the rule that fired.",
    ["rule"],
)
risk_score = Histogram(
    "fraud_risk_score",
    "The model's fraud probability for each scored transaction.",
    ["source"],
    # The thresholds (0.24, 0.95) fall on bucket edges, so the share of
    # transactions above each one can be read straight off the histogram.
    buckets=(0.001, 0.01, 0.05, 0.24, 0.5, 0.95, 1.0),
)

# ------------------------------------------------------------------ velocity
velocity_duration_seconds = Histogram(
    "fraud_velocity_duration_seconds",
    "Time to measure a card's recent activity in Redis, per call.",
    buckets=VELOCITY_BUCKETS,
)
velocity_failures_total = Counter(
    "fraud_velocity_failures_total",
    "Calls that could not reach Redis, by whether Redis was being left alone.",
    ["reason"],
)
velocity_skipped_total = Counter(
    "fraud_velocity_skipped_total",
    "Transactions scored without velocity features because Redis was unavailable.",
)

# ------------------------------------------------------------------ consumer
messages_total = Counter(
    "fraud_consumer_messages_total",
    "Messages read from Kafka, by what happened to them.",
    ["outcome"],  # scored, dead_lettered, duplicate
)
batch_size = Histogram(
    "fraud_consumer_batch_size",
    "Messages per batch. A consumer keeping up reads small batches.",
    buckets=(1, 5, 10, 50, 100, 250, 500, 1000),
)
batch_duration_seconds = Histogram(
    "fraud_consumer_batch_duration_seconds",
    "Time to decode, score, record and commit one batch.",
    buckets=LATENCY_BUCKETS,
)
consumer_lag = Gauge(
    "fraud_consumer_lag",
    "Messages in a partition the consumer has not read yet.",
    ["partition"],
)
consumer_partitions = Gauge(
    "fraud_consumer_partitions",
    "Partitions currently assigned to this consumer.",
)
consumer_last_poll_timestamp = Gauge(
    "fraud_consumer_last_poll_timestamp_seconds",
    "When the consumer last polled Kafka. Stops moving if the loop is stuck.",
)
store_outages_total = Counter(
    "fraud_store_outages_total",
    "Times the decision store was found to be unavailable while consuming.",
)
store_outage_seconds_total = Counter(
    "fraud_store_outage_seconds_total",
    "Seconds spent waiting for the decision store to come back.",
)

# --------------------------------------------------------------------- model
model_loaded = Gauge(
    "fraud_model_loaded",
    "1 for the model version this process is serving.",
    ["model_version"],
)


def record_results(results: Sequence[RiskResult], source: str) -> None:
    """Count what was decided. `source` is "api" or "stream"."""
    for result in results:
        decisions_total.labels(decision=result.decision.value, source=source).inc()
        risk_score.labels(source=source).observe(result.risk_score)
        if result.model_decision is not None and result.decision is not result.model_decision:
            for rule in result.reasons:
                escalations_total.labels(rule=rule).inc()
