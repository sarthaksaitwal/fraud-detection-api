"""Score transactions from Kafka and record the decisions (Steps 4.4, 4.5).

For every batch read from the topic:

    1. decode      each message with the API's Transaction schema
    2. dead-letter the ones that fail, with the reason, to KAFKA_DLQ_TOPIC
    3. deduplicate transaction ids (Postgres refuses one id twice in one save)
    4. measure     what each card has done recently   -- src/features/velocity.py
    5. score       the batch with one model call      -- src/scoring.py, as /score/batch
    6. record      the decisions with one INSERT       -- src/storage/store.py, as /score/batch
    7. commit      the batch's offsets, only now

Committing last makes delivery at-least-once. A crash anywhere before step 7
means the batch is read again on restart: nothing is lost, and the repeat is
harmless because recording is an upsert keyed by transaction_id, and a repeated
velocity measurement re-adds the same member rather than counting it twice.

If the database is down at step 6, the consumer waits instead of dropping the
batch or crashing (Step 4.5). It pauses its partitions, so no new messages are
fetched, and keeps polling Kafka, so the group does not decide it has died and
hand its partitions to someone else. It retries with growing delays until the
decisions are recorded, then commits and resumes. Messages keep arriving in
Kafka meanwhile, and are worked through once the database is back.

This is the difference from POST /score, which has a customer waiting and so
answers without recording. Nobody waits on the consumer, so it can wait for
the database, and no decision is lost.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import CommitFailedError
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import Decision, RiskResult, Transaction, VelocityFeatures
from src.config import settings
from src.features.redis_client import open_configured_redis
from src.features.velocity import VelocityStore, open_configured_velocity
from src.scoring import Scorer
from src.storage.db import create_db_engine
from src.storage.store import DecisionStore, failure_reason
from src.streaming.kafka import check_topics
from src.streaming.messages import (
    REASON_HEADER,
    InvalidMessageError,
    KafkaRecord,
    dead_letter,
    decode_transaction,
    header,
)

log = logging.getLogger("src.streaming.consumer")

POLL_TIMEOUT_MS = 1000
PROGRESS_EVERY_SECONDS = 10.0
# Waits between attempts to record a batch while the database is down.
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0, 8.0, 15.0)


class Message(Protocol):
    """The parts of a Kafka ConsumerRecord the consumer uses."""

    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None


@dataclass
class PreparedBatch:
    """A batch after decoding, before scoring."""

    transactions: list[Transaction]
    dead_letters: list[KafkaRecord]
    duplicates: int


@dataclass
class ConsumeStats:
    consumed: int = 0
    scored: int = 0
    dead_lettered: int = 0
    duplicates: int = 0
    batches: int = 0
    decisions: Counter[Decision] = field(default_factory=Counter)
    outages: int = 0
    seconds_waiting: float = 0.0
    seconds: float = 0.0
    # Set when stopped during an outage: the batch was not recorded or committed.
    interrupted: bool = False

    def add(self, other: ConsumeStats) -> None:
        self.consumed += other.consumed
        self.scored += other.scored
        self.dead_lettered += other.dead_lettered
        self.duplicates += other.duplicates
        self.batches += other.batches
        self.decisions.update(other.decisions)
        self.outages += other.outages
        self.seconds_waiting += other.seconds_waiting

    def summary(self) -> str:
        rate = self.consumed / self.seconds if self.seconds else 0.0
        decided = ", ".join(f"{self.decisions[d]} {d.value}" for d in Decision)
        waited = (
            f"; waited {self.seconds_waiting:.0f}s for the database in {self.outages} outage(s)"
            if self.outages
            else ""
        )
        return (
            f"consumed {self.consumed} message(s) in {self.batches} batch(es), {self.seconds:.1f}s "
            f"({rate:.0f}/s): scored {self.scored} ({decided}), "
            f"{self.dead_lettered} dead-lettered, {self.duplicates} duplicate(s) skipped{waited}"
        )


def prepare_batch(messages: Sequence[Message]) -> PreparedBatch:
    """Decode messages, turning bad ones into dead letters and dropping repeated ids.

    When an id repeats within a batch, the latest message wins, as it would if
    the two had arrived in separate batches and the second upsert replaced the first.
    """
    latest: dict[str, Transaction] = {}
    dead_letters, valid = [], 0
    for message in messages:
        try:
            transaction = decode_transaction(message.key, message.value)
        except InvalidMessageError as error:
            dead_letters.append(
                dead_letter(
                    message.key,
                    message.value,
                    error.reason,
                    message.topic,
                    message.partition,
                    message.offset,
                )
            )
            continue
        valid += 1
        latest.pop(transaction.transaction_id, None)  # re-insert so order follows the latest
        latest[transaction.transaction_id] = transaction
    return PreparedBatch(list(latest.values()), dead_letters, valid - len(latest))


def offsets_to_commit(
    batches: Mapping[TopicPartition, Sequence[Message]],
) -> dict[TopicPartition, int]:
    """Per partition, the offset after the last message read: where to resume."""
    return {tp: messages[-1].offset + 1 for tp, messages in batches.items() if messages}


async def wait_polling(
    consumer: Any,
    seconds: float,
    stop: asyncio.Event | None,
    clock: Callable[[], float],
) -> None:
    """Wait `seconds` while still polling, so Kafka knows the consumer is alive.

    The partitions are paused, so polls should return nothing. A rebalance can
    un-pause newly assigned partitions, though; anything fetched then is put back
    by seeking to its first offset, or the consumer would move past it unread.
    """
    deadline = clock() + seconds
    while clock() < deadline and not (stop is not None and stop.is_set()):
        remaining_ms = max(1, int((deadline - clock()) * 1000))
        fetched = await consumer.getmany(timeout_ms=min(POLL_TIMEOUT_MS, remaining_ms))
        for tp, messages in fetched.items():
            if messages:
                consumer.seek(tp, messages[0].offset)


async def record_until_done(
    results: Sequence[RiskResult],
    consumer: Any,
    store: DecisionStore,
    stats: ConsumeStats,
    stop: asyncio.Event | None,
    clock: Callable[[], float],
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> bool:
    """Record the decisions, waiting out a database outage. False if stopped first."""
    if await asyncio.to_thread(store.record, results, True):
        return True

    stats.outages += 1
    started = clock()
    paused = consumer.assignment()
    consumer.pause(*paused)
    log.warning(
        "decision store unavailable; paused %d partition(s) and holding %d decision(s) "
        "until it is back",
        len(paused),
        len(results),
    )
    try:
        attempt = 0
        while not (stop is not None and stop.is_set()):
            delay = retry_delays[min(attempt, len(retry_delays) - 1)]
            await wait_polling(consumer, delay, stop, clock)
            if stop is not None and stop.is_set():
                break
            attempt += 1
            if await asyncio.to_thread(store.record, results, True):
                log.warning(
                    "decision store is back after %.0fs and %d retries; resuming",
                    clock() - started,
                    attempt,
                )
                return True
        log.warning(
            "stopping during a database outage; the uncommitted batch will be read again "
            "on restart"
        )
        return False
    finally:
        stats.seconds_waiting += clock() - started
        # A rebalance may have changed the assignment while waiting.
        consumer.resume(*consumer.assignment())


async def measure(
    velocity: VelocityStore | None, transactions: Sequence[Transaction]
) -> list[VelocityFeatures | None]:
    """Recent activity per card, in a worker thread: Redis blocks, this loop must not.

    Velocity never fails a batch. Redis being down costs the features, which is
    exactly the opposite of the database, which the consumer waits for.
    """
    if velocity is None:
        return [None] * len(transactions)
    return await asyncio.to_thread(velocity.measure_or_none, transactions)


async def handle_batch(
    batches: Mapping[TopicPartition, Sequence[Message]],
    consumer: Any,
    dlq_producer: Any,
    dlq_topic: str,
    scorer: Scorer,
    store: DecisionStore,
    velocity: VelocityStore | None = None,
    stop: asyncio.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> ConsumeStats:
    """Steps 1-7 for one batch.

    If stopped during an outage, nothing is committed and the returned stats say
    only how long it waited, with `interrupted` set: the batch will be read again.
    """
    messages = [message for partition in batches.values() for message in partition]
    prepared = prepare_batch(messages)
    stats = ConsumeStats(
        consumed=len(messages),
        scored=len(prepared.transactions),
        dead_lettered=len(prepared.dead_letters),
        duplicates=prepared.duplicates,
        batches=1,
    )

    for record in prepared.dead_letters:
        # Waits for Kafka's confirmation: a dead letter must be stored before the
        # original message's offset is committed, or the message would be lost.
        await dlq_producer.send_and_wait(
            dlq_topic, record.value, key=record.key, headers=list(record.headers)
        )
        log.warning(
            "dead-lettered %s: %s",
            (record.key or b"<no key>").decode("utf-8", "replace"),
            header(record.headers, REASON_HEADER),
        )

    if prepared.transactions:
        # Scoring is CPU work and recording blocks on the database, so both run in a
        # worker thread; the event loop stays free to keep the Kafka session alive.
        features = await measure(velocity, prepared.transactions)
        results = await asyncio.to_thread(scorer.score, prepared.transactions, features)
        if not await record_until_done(results, consumer, store, stats, stop, clock, retry_delays):
            return ConsumeStats(
                outages=stats.outages, seconds_waiting=stats.seconds_waiting, interrupted=True
            )
        stats.decisions.update(result.decision for result in results)

    try:
        await consumer.commit(offsets_to_commit(batches))
    except CommitFailedError as error:
        # The group rebalanced and these partitions now belong to another consumer,
        # which will read the batch again. Recording is an upsert, so that is safe.
        log.warning("commit failed after a rebalance; the batch will be re-read: %s", error)
    return stats


async def total_lag(consumer: Any) -> int | None:
    """Messages in the assigned partitions that this consumer has not yet read."""
    total = 0
    for tp in consumer.assignment():
        highwater = consumer.highwater(tp)
        if highwater is None:
            return None
        total += highwater - await consumer.position(tp)
    return total


async def consume(
    consumer: Any,
    dlq_producer: Any,
    scorer: Scorer,
    store: DecisionStore,
    dlq_topic: str,
    max_batch: int,
    stop: asyncio.Event | None = None,
    until_idle: float | None = None,
    velocity: VelocityStore | None = None,
    clock: Callable[[], float] = time.monotonic,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> ConsumeStats:
    """Handle batches until `stop` is set, or until nothing arrives for `until_idle` seconds."""
    stats = ConsumeStats()
    started = last_progress = last_message = clock()
    failed = True
    try:
        while stop is None or not stop.is_set():
            batches = await consumer.getmany(timeout_ms=POLL_TIMEOUT_MS, max_records=max_batch)
            if any(batches.values()):
                handled = await handle_batch(
                    batches,
                    consumer,
                    dlq_producer,
                    dlq_topic,
                    scorer,
                    store,
                    velocity=velocity,
                    stop=stop,
                    clock=clock,
                    retry_delays=retry_delays,
                )
                stats.add(handled)
                if handled.interrupted:
                    break
                last_message = clock()
            elif until_idle is not None and clock() - last_message >= until_idle:
                log.info("no messages for %.0fs; stopping", until_idle)
                break
            if clock() - last_progress >= PROGRESS_EVERY_SECONDS:
                last_progress = clock()
                stats.seconds = last_progress - started
                log.info("%s; lag %s", stats.summary(), await total_lag(consumer))
        failed = False
    finally:
        stats.seconds = clock() - started
        if failed:
            log.error("stopped by an error after: %s", stats.summary())
    return stats


async def run(
    stop: asyncio.Event | None = None,
    until_idle: float | None = None,
    group: str | None = None,
    topic: str | None = None,
    dlq_topic: str | None = None,
    bootstrap_servers: str | None = None,
    max_batch: int | None = None,
    scorer: Scorer | None = None,
    store: DecisionStore | None = None,
    velocity: VelocityStore | None = None,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> ConsumeStats:
    """Load the model, connect to Kafka, Postgres and Redis, consume, and always disconnect."""
    topic = topic or settings.kafka_topic
    dlq_topic = dlq_topic or settings.kafka_dlq_topic
    group = group or settings.kafka_consumer_group
    bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
    max_batch = max_batch or settings.consumer_max_batch

    await check_topics([topic, dlq_topic], bootstrap_servers)
    scorer = scorer or Scorer.load()
    owns_store = store is None
    store = store or DecisionStore(create_db_engine())
    try:
        store.ensure_schema()
    except SQLAlchemyError as exc:
        # Not fatal: the first batch waits for the database like any later one.
        log.warning("decision store unavailable at startup: %s", failure_reason(exc))

    # Redis is optional in a way Postgres is not: without it, transactions are
    # scored without velocity features rather than waiting.
    redis = None
    if velocity is None:
        redis = open_configured_redis()
        velocity = open_configured_velocity(redis)

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group,
        enable_auto_commit=False,  # offsets are committed by hand, after recording
        auto_offset_reset="earliest",  # a new group starts from the oldest message
        max_poll_records=max_batch,
    )
    dlq_producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers, acks="all", enable_idempotence=True
    )
    await consumer.start()
    await dlq_producer.start()
    log.info("consuming %s as group %s, up to %d per batch", topic, group, max_batch)
    try:
        return await consume(
            consumer,
            dlq_producer,
            scorer,
            store,
            dlq_topic,
            max_batch,
            stop,
            until_idle,
            velocity=velocity,
            retry_delays=retry_delays,
        )
    finally:
        await dlq_producer.stop()
        await consumer.stop()  # leaves the group so partitions are reassigned at once
        if owns_store:
            store.close()
        if redis is not None:
            redis.close()
