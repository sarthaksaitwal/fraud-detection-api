"""Step 6.5 guard rails: the load test measures what it claims to.

The stages run against the app in this process through httpx's ASGI transport,
so the tool is exercised end to end without a server or a network.
"""

import asyncio

import httpx
import numpy as np
import pytest

from scripts import load_test as tool
from scripts.load_test import StageResult, knee, percentile, run_stage
from src.api.main import create_app
from src.ml.preprocess import RAW_FEATURES
from src.scoring import Scorer


class AmountAsProbability:
    """Stands in for the model: fraud probability = Amount / 100, capped at 1."""

    def predict_proba(self, X):
        p = np.clip(X["Amount"].to_numpy() / 100, 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def bodies(raw_df):
    """Request bodies, as the tool builds them from the test split."""
    return raw_df[RAW_FEATURES].head(5).to_dict(orient="records")


@pytest.fixture
def client():
    scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
    app = create_app(load_scorer=lambda: scorer)

    async def build():
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://api")

    return asyncio.run(build())


def stage(client, bodies, concurrency=2, seconds=0.2):
    async def go():
        async with client:
            # The app's lifespan is not run by ASGITransport, so the state the
            # routes read is set here instead.
            client._transport.app.state.scorer = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
            client._transport.app.state.store = None
            client._transport.app.state.velocity = None
            return await run_stage(client, bodies, concurrency, seconds)

    return asyncio.run(go())


# ------------------------------------------------------------------ percentiles
def test_a_percentile_is_always_a_measured_value():
    """Nearest rank, not interpolation: p99 is a request someone really waited for."""
    values = [float(n) for n in range(1, 101)]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 99) == 99


def test_percentiles_of_one_value_and_of_none():
    assert percentile([0.5], 99) == 0.5
    assert percentile([], 50) == 0.0


def test_percentiles_do_not_care_about_order():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0


# ----------------------------------------------------------------- the stages
def test_a_stage_keeps_every_caller_busy(client, bodies):
    result = stage(client, bodies, concurrency=3, seconds=0.3)
    assert result.requests > 3  # each caller sent more than one
    assert result.errors == 0
    assert len(result.latencies) == result.requests


def test_a_stage_reports_throughput_and_percentiles(client, bodies):
    result = stage(client, bodies, concurrency=2, seconds=0.3)
    assert result.throughput == pytest.approx(result.requests / result.seconds)
    assert result.p50 <= result.p95 <= result.p99


def test_every_request_carries_its_own_transaction_id(client, bodies):
    """Reusing one id would make the database update a row instead of inserting."""
    result = stage(client, bodies, concurrency=2, seconds=0.3)
    assert result.requests > 1
    # The ids are built from worker and sequence number, so none can repeat.
    ids = {f"load-{worker}-{sent}" for worker in range(2) for sent in range(result.requests)}
    assert len(ids) == 2 * result.requests


def test_a_stage_survives_a_service_that_refuses(bodies):
    """A load test that crashes when the service does tells you nothing."""

    async def go():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:1", timeout=1.0) as client:
            return await run_stage(client, bodies, concurrency=2, seconds=0.2)

    result = asyncio.run(go())
    assert result.requests == 0
    assert result.errors > 0


def test_results_are_json_without_the_raw_latencies():
    result = StageResult(concurrency=4, seconds=10.0, requests=100)
    result.latencies = [0.01] * 100
    summary = result.as_dict()
    assert "latencies" not in summary
    assert summary["p50_ms"] == 10.0
    assert summary["throughput"] == 10.0


# ------------------------------------------------------------------- the knee
def result(concurrency, throughput, p95):
    """A stage with the throughput and p95 a test wants to describe."""
    made = StageResult(concurrency=concurrency, seconds=1.0, requests=int(throughput))
    made.latencies = [p95] * int(throughput)
    return made


def test_the_knee_is_where_throughput_stops_paying_for_latency():
    results = [result(1, 100, 0.01), result(2, 190, 0.011), result(4, 195, 0.05)]
    assert knee(results).concurrency == 2


def test_a_service_that_keeps_up_has_no_knee():
    results = [result(1, 100, 0.01), result(2, 200, 0.011), result(4, 400, 0.012)]
    assert knee(results) is None


def test_the_knee_of_a_single_stage_is_nothing():
    assert knee([result(1, 100, 0.01)]) is None


# ------------------------------------------------------------------ the script
@pytest.mark.parametrize(
    "args", [["--concurrency", "many"], ["--seconds", "0"], ["--concurrency", "0"]]
)
def test_the_script_rejects_bad_arguments(args):
    with pytest.raises(SystemExit):
        tool.main(args)


def test_the_script_says_so_when_nothing_answers(capsys):
    assert (
        tool.main(
            [
                "--url",
                "http://127.0.0.1:1",
                "--concurrency",
                "1",
                "--seconds",
                "0.2",
                "--warmup",
                "0",
            ]
        )
        == 1
    )
    assert "is the API running?" in capsys.readouterr().err
