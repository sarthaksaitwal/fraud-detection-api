"""Drive POST /score until latency turns upward (Step 6.5).

    python -m scripts.load_test                                   # 1 → 64 concurrent callers
    python -m scripts.load_test --concurrency 1,8,32 --seconds 5
    python -m scripts.load_test --url http://127.0.0.1:8000 --out data/load-test.json

Each stage keeps a fixed number of callers busy for `--seconds`: every caller
sends a request, waits for the answer, and sends the next one. That is what a
queue of real clients does, and unlike firing a fixed rate at the server it
cannot pretend the server is keeping up when it is not.

What it reports, per stage:

    throughput   answered requests per second
    p50/p95/p99  what a caller waited, including the network
    server       what the server says it spent, from its own histogram

The gap between the last two is queueing: time a request spent waiting for a
worker thread, a database connection or the event loop, rather than being
worked on. Watching the two diverge is the point of the exercise.

Transactions come from the test split with their synthetic cards, so velocity
and the rules do the work they would really do. Every request gets a fresh
transaction id, so the database inserts rather than updating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from src.streaming.producer import load_transactions

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
# Bodies are reused in rotation; enough of them that the model never sees the
# same transaction twice in a row and caches cannot flatter the result.
BODY_POOL = 200


@dataclass
class StageResult:
    concurrency: int
    seconds: float
    requests: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list, repr=False)
    server_seconds: float | None = None  # from the API's own histogram

    @property
    def throughput(self) -> float:
        return self.requests / self.seconds if self.seconds else 0.0

    @property
    def p50(self) -> float:
        return percentile(self.latencies, 50)

    @property
    def p95(self) -> float:
        return percentile(self.latencies, 95)

    @property
    def p99(self) -> float:
        return percentile(self.latencies, 99)

    def row(self) -> str:
        server = f"{self.server_seconds * 1000:7.1f}" if self.server_seconds else "      -"
        return (
            f"{self.concurrency:>11} {self.throughput:>9.1f} {self.p50 * 1000:>8.1f} "
            f"{self.p95 * 1000:>8.1f} {self.p99 * 1000:>8.1f} {server} {self.errors:>7}"
        )

    def as_dict(self) -> dict:
        summary = asdict(self)
        summary.pop("latencies")
        return summary | {
            "throughput": round(self.throughput, 1),
            "p50_ms": round(self.p50 * 1000, 2),
            "p95_ms": round(self.p95 * 1000, 2),
            "p99_ms": round(self.p99 * 1000, 2),
        }


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile by nearest rank, so it is always a measured value."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p / 100 * len(ordered))))
    return ordered[rank - 1]


def request_bodies(count: int = BODY_POOL) -> list[dict]:
    """Real transactions, with the cards and countries the producer would send."""
    return [transaction.model_dump(mode="json") for transaction in load_transactions(limit=count)]


async def caller(
    client: httpx.AsyncClient,
    bodies: list[dict],
    deadline: float,
    result: StageResult,
    worker: int,
    clock=time.perf_counter,
) -> None:
    """One caller: request, wait, request again, until the stage is over."""
    sent = 0
    while clock() < deadline:
        body = dict(bodies[(worker + sent) % len(bodies)])
        body["transaction_id"] = f"load-{worker}-{sent}"
        sent += 1
        started = clock()
        try:
            response = await client.post("/score", json=body)
            elapsed = clock() - started
        except httpx.HTTPError:
            result.errors += 1
            continue
        if response.status_code != 200:
            result.errors += 1
            continue
        result.requests += 1
        result.latencies.append(elapsed)


async def server_mean_seconds(client: httpx.AsyncClient) -> tuple[float, int] | None:
    """The API's own total time and request count for /score, from /metrics.

    Two scrapes bracket a stage; the difference is what the server thinks it
    spent on that stage alone.
    """
    try:
        response = await client.get("/metrics")
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    total = count = None
    for line in response.text.splitlines():
        if line.startswith("fraud_http_request_duration_seconds_sum") and '"/score"' in line:
            total = float(line.rsplit(" ", 1)[1])
        elif line.startswith("fraud_http_request_duration_seconds_count") and '"/score"' in line:
            count = float(line.rsplit(" ", 1)[1])
    return (total, count) if total is not None and count is not None else None


async def run_stage(
    client: httpx.AsyncClient,
    bodies: list[dict],
    concurrency: int,
    seconds: float,
    clock=time.perf_counter,
) -> StageResult:
    """Keep `concurrency` callers busy for `seconds`, and report what happened."""
    result = StageResult(concurrency=concurrency, seconds=seconds)
    before = await server_mean_seconds(client)
    started = clock()
    await asyncio.gather(
        *(
            caller(client, bodies, started + seconds, result, worker, clock)
            for worker in range(concurrency)
        )
    )
    result.seconds = clock() - started
    after = await server_mean_seconds(client)
    if before and after and after[1] > before[1]:
        result.server_seconds = (after[0] - before[0]) / (after[1] - before[1])
    return result


async def load_test(
    url: str, concurrency: list[int], seconds: float, warmup: float
) -> list[StageResult]:
    bodies = request_bodies()
    results = []
    # One connection per caller, kept open: reconnecting on every request would
    # measure the TCP handshake as much as the service.
    limits = httpx.Limits(
        max_connections=max(concurrency), max_keepalive_connections=max(concurrency)
    )
    async with httpx.AsyncClient(base_url=url, timeout=30.0, limits=limits) as client:
        if warmup:
            print(f"warming up for {warmup:g}s", file=sys.stderr)
            await run_stage(client, bodies, 2, warmup)
        print(" concurrency throughput      p50      p95      p99  server  errors")
        for level in concurrency:
            result = await run_stage(client, bodies, level, seconds)
            results.append(result)
            print(result.row(), flush=True)
    return results


def knee(results: list[StageResult]) -> StageResult | None:
    """The first stage where throughput stops rising but latency keeps climbing.

    Past it, more callers only mean longer queues: the service is saturated.
    """
    for earlier, later in zip(results, results[1:], strict=False):
        if later.throughput < earlier.throughput * 1.1 and later.p95 > earlier.p95 * 1.5:
            return earlier
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.load_test",
        description="Drive POST /score at rising concurrency and report where latency turns.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"default: {DEFAULT_URL}")
    parser.add_argument(
        "--concurrency",
        default=",".join(str(level) for level in DEFAULT_CONCURRENCY),
        help="comma-separated caller counts, one stage each",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="per stage; default 10")
    parser.add_argument("--warmup", type=float, default=3.0, help="discarded; default 3")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args(argv)

    try:
        concurrency = [int(level) for level in args.concurrency.split(",")]
    except ValueError:
        parser.error("--concurrency must be comma-separated whole numbers, e.g. 1,8,32")
    if any(level < 1 for level in concurrency):
        parser.error("--concurrency levels must be at least 1")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    results = asyncio.run(load_test(args.url, concurrency, args.seconds, args.warmup))
    if not any(result.requests for result in results):
        print(f"no requests answered at {args.url}: is the API running?", file=sys.stderr)
        return 1

    turned = knee(results)
    if turned is None:
        print("\nLatency never turned: the service kept up at every level tried.")
    else:
        print(
            f"\nLatency turns after {turned.concurrency} concurrent caller(s): "
            f"{turned.throughput:.0f}/s at p95 {turned.p95 * 1000:.0f}ms."
        )
    errors = sum(result.errors for result in results)
    if errors:
        print(f"{errors} request(s) failed; see the API log.", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([result.as_dict() for result in results], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
