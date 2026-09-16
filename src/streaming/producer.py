"""Send test-set transactions to Kafka, standing in for a card network (Step 4.3).

Transactions go out in the order they happened (by `Time`), keyed by
transaction_id, at a steady rate. Their ids are `test-row-<row>`, the same ids
scripts/sample_request.py uses, so a streamed transaction can be looked up with
GET /transactions/test-row-<row> once the consumer has scored it.

Delivery guarantees:
  acks="all"                 Kafka confirms each message only once it is stored
  enable_idempotence=True    a retried send is stored once, not twice
A message counts as sent only after Kafka confirms it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from aiokafka import AIOKafkaProducer

from src.api.schemas import Transaction
from src.config import settings
from src.ml.preprocess import RAW_FEATURES
from src.streaming.kafka import check_topics
from src.streaming.messages import encode_transaction

log = logging.getLogger("src.streaming.producer")

# Confirmations are collected in groups this size, so memory stays flat and a
# failed send is noticed within one group rather than at the very end.
CONFIRM_EVERY = 500
PROGRESS_EVERY_SECONDS = 10.0


def load_transactions(
    start: int = 0, limit: int | None = None, path: Path | None = None
) -> list[Transaction]:
    """Test-set transactions in time order, from position `start`, at most `limit` of them."""
    test = pd.read_parquet(path or settings.test_split_path)
    # A stable sort, with the row number breaking ties, so positions never change.
    ordered = test.rename_axis("row").sort_values(["Time", "row"], kind="stable")
    if start >= len(ordered):
        raise ValueError(f"--start {start} is past the end: the test set has {len(ordered)} rows")
    chosen = ordered.iloc[start : None if limit is None else start + limit]
    return [
        Transaction(transaction_id=f"test-row-{row}", **values)
        for row, values in zip(chosen.index, chosen[RAW_FEATURES].to_dict("records"), strict=True)
    ]


@dataclass
class ProduceReport:
    """Progress so far. Kept up to date while sending, so it is right even if stopped early."""

    start: int
    sent: int = 0
    partitions: Counter[int] = field(default_factory=Counter)
    seconds: float = 0.0

    @property
    def next_start(self) -> int:
        """The --start that resumes exactly after the last confirmed message."""
        return self.start + self.sent

    @property
    def rate(self) -> float:
        return self.sent / self.seconds if self.seconds else 0.0

    def summary(self, topic: str) -> str:
        spread = ", ".join(f"p{p}: {n}" for p, n in sorted(self.partitions.items()))
        return (
            f"sent {self.sent} transaction(s) to {topic} in {self.seconds:.1f}s "
            f"({self.rate:.1f}/s; {spread or 'none'}); resume with --start {self.next_start}"
        )


async def produce(
    transactions: Iterable[Transaction],
    producer: AIOKafkaProducer,
    topic: str,
    rate: float,
    report: ProduceReport,
    stop: asyncio.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ProduceReport:
    """Send transactions at `rate` per second (0 = as fast as possible) until done or `stop`.

    Pacing follows a fixed schedule -- message i is due at start + i / rate --
    rather than sleeping 1 / rate after each send. Sleeping would add every
    send's own time on top, and the stream would drift slower than asked.
    """
    started = clock()
    last_progress = started
    pending: list[asyncio.Future] = []

    async def confirm() -> None:
        # One at a time, in order, so a failure leaves `sent` counting exactly the
        # messages confirmed before it. The error then stops the whole run.
        for future in pending:
            metadata = await future
            report.sent += 1
            report.partitions[metadata.partition] += 1
        pending.clear()

    try:
        for position, transaction in enumerate(transactions):
            if stop is not None and stop.is_set():
                break
            if rate > 0 and started + position / rate > clock():
                # Ahead of schedule: collect confirmations while waiting anyway. At a
                # slow rate that confirms every message; at a fast one, whole batches.
                await confirm()
                delay = started + position / rate - clock()
                if delay > 0:
                    await sleep(delay)
            record = encode_transaction(transaction)
            # send() queues the message and returns a future that completes when
            # Kafka confirms it; producers batch queued messages into one request.
            pending.append(await producer.send(topic, record.value, key=record.key))
            if len(pending) >= CONFIRM_EVERY:
                await confirm()
            if clock() - last_progress >= PROGRESS_EVERY_SECONDS:
                last_progress = clock()
                report.seconds = last_progress - started
                log.info("sent %d so far (%.1f/s)", report.sent, report.rate)
        await confirm()
    finally:
        report.seconds = clock() - started
    return report


async def run(
    start: int = 0,
    limit: int | None = None,
    rate: float | None = None,
    topic: str | None = None,
    bootstrap_servers: str | None = None,
    stop: asyncio.Event | None = None,
    transactions: Sequence[Transaction] | None = None,
) -> ProduceReport:
    """Check the topic, connect, send, and always flush and disconnect."""
    topic = topic or settings.kafka_topic
    bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
    rate = settings.producer_rate_per_sec if rate is None else rate
    if transactions is None:
        transactions = load_transactions(start, limit)

    await check_topics([topic], bootstrap_servers)
    producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        acks="all",
        enable_idempotence=True,
        linger_ms=5,  # wait up to 5 ms to fill a batch: far fewer requests at full speed
    )
    await producer.start()
    report = ProduceReport(start=start)
    log.info(
        "sending %d transaction(s) to %s at %s",
        len(transactions),
        topic,
        f"{rate:g}/s" if rate > 0 else "full speed",
    )
    try:
        await produce(transactions, producer, topic, rate, report, stop)
    except BaseException:
        log.error("stopped by an error: %s", report.summary(topic))
        raise
    finally:
        await producer.stop()  # sends anything still queued before disconnecting
    return report
