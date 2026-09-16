"""Steps 4.4 and 4.5 guard rails: the consumer."""

import asyncio
import logging
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import CommitFailedError

from scripts import consume as consume_script
from src.api.schemas import Decision, Transaction
from src.ml.preprocess import RAW_FEATURES
from src.scoring import Scorer
from src.storage.db import create_db_engine
from src.storage.store import DecisionStore
from src.streaming import consumer as consumer_module
from src.streaming.consumer import (
    ConsumeStats,
    consume,
    handle_batch,
    offsets_to_commit,
    prepare_batch,
    record_until_done,
    run,
    wait_polling,
)
from src.streaming.messages import (
    REASON_HEADER,
    SOURCE_OFFSET_HEADER,
    encode_transaction,
    header,
)


class AmountAsProbability:
    """Stands in for the model: fraud probability = Amount / 100, capped at 1."""

    def predict_proba(self, X):
        p = np.clip(X["Amount"].to_numpy() / 100, 0, 1)
        return np.column_stack([1 - p, p])


SCORER = Scorer(AmountAsProbability(), "v-test", 0.24, 0.95)
TP0, TP1 = TopicPartition("transactions", 0), TopicPartition("transactions", 1)


@pytest.fixture
def make_transaction(raw_df):
    row = raw_df[RAW_FEATURES].iloc[0].to_dict()

    def make(transaction_id, amount=10.0):
        return Transaction(**{**row, "transaction_id": transaction_id, "Amount": amount})

    return make


def message(transaction=None, offset=0, partition=0, key=None, value=None):
    """A Kafka record, from a transaction or from raw bytes."""
    if transaction is not None:
        record = encode_transaction(transaction)
        key, value = record.key, record.value
    return SimpleNamespace(
        topic="transactions", partition=partition, offset=offset, key=key, value=value
    )


@pytest.fixture
def store(tmp_path):
    decision_store = DecisionStore(create_db_engine(f"sqlite:///{(tmp_path / 'd.db').as_posix()}"))
    yield decision_store
    decision_store.close()


class FlakyStore:
    """Refuses the first `failures` batches (None = every batch), then passes them on."""

    def __init__(self, store=None, failures=None):
        self.store, self.failures = store, failures
        self.calls = []

    def ensure_schema(self):
        pass

    def record(self, results, will_retry=False):
        self.calls.append(will_retry)
        if self.failures is None or len(self.calls) <= self.failures:
            return False
        return self.store.record(results, will_retry)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeConsumer:
    """Records pauses, resumes, seeks and commits. Empty polls wait out their timeout."""

    def __init__(self, batches=(), commit_error=None, clock=None, strays=()):
        self.batches = list(batches)
        self.strays = list(strays)  # returned while paused, as after a rebalance
        self.commits, self.seeks, self.events = [], [], []
        self.commit_error = commit_error
        self.clock = clock or FakeClock()
        self.paused = set()

    async def getmany(self, timeout_ms=0, max_records=None):
        await asyncio.sleep(0)  # a real poll always hands control back to the event loop
        if self.paused:
            self.clock.now += timeout_ms / 1000
            return self.strays.pop(0) if self.strays else {}
        if self.batches:
            return self.batches.pop(0)
        self.clock.now += timeout_ms / 1000
        return {}

    async def commit(self, offsets):
        if self.commit_error:
            raise self.commit_error
        self.events.append("commit")
        self.commits.append(offsets)

    def assignment(self):
        return {TP0, TP1}

    def pause(self, *partitions):
        self.events.append("pause")
        self.paused.update(partitions)

    def resume(self, *partitions):
        self.events.append("resume")
        self.paused.difference_update(partitions)

    def seek(self, tp, offset):
        self.seeks.append((tp, offset))

    def highwater(self, tp):
        return None


class FakeProducer:
    def __init__(self, events=None):
        self.sent = []
        self.events = events

    async def send_and_wait(self, topic, value, key=None, headers=None):
        self.sent.append((topic, key, value, headers))
        if self.events is not None:
            self.events.append("dead-letter")


async def stop_soon(coroutine_factory, after=0.05):
    """Run a coroutine with a stop event that is set shortly after it starts."""
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(after, stop.set)
    return await coroutine_factory(stop)


# ------------------------------------------------------------ preparing a batch
def test_valid_messages_are_decoded_in_order(make_transaction):
    transactions = [make_transaction("tx-1"), make_transaction("tx-2")]
    prepared = prepare_batch([message(t, offset=i) for i, t in enumerate(transactions)])
    assert prepared.transactions == transactions
    assert (prepared.dead_letters, prepared.duplicates) == ([], 0)


def test_bad_messages_become_dead_letters_that_say_where_they_came_from(make_transaction):
    prepared = prepare_batch(
        [
            message(make_transaction("tx-1"), offset=7),
            message(key=b"tx-2", value=b"not json", offset=8, partition=2),
        ]
    )
    assert [t.transaction_id for t in prepared.transactions] == ["tx-1"]
    (dead,) = prepared.dead_letters
    assert (dead.key, dead.value) == (b"tx-2", b"not json")
    assert "Invalid JSON" in header(dead.headers, REASON_HEADER)
    assert header(dead.headers, SOURCE_OFFSET_HEADER) == "8"


def test_a_repeated_id_is_scored_once_using_its_latest_message(make_transaction):
    prepared = prepare_batch(
        [
            message(make_transaction("tx-1", amount=10), offset=0),
            message(make_transaction("tx-2"), offset=1),
            message(make_transaction("tx-1", amount=99), offset=2),
        ]
    )
    assert [t.transaction_id for t in prepared.transactions] == ["tx-2", "tx-1"]
    assert prepared.transactions[1].Amount == 99
    assert prepared.duplicates == 1


def test_offsets_to_commit_are_one_past_the_last_message_per_partition(make_transaction):
    t = make_transaction("tx")
    batches = {TP0: [message(t, offset=4), message(t, offset=5)], TP1: [message(t, offset=9)]}
    assert offsets_to_commit(batches) == {TP0: 6, TP1: 10}


# ------------------------------------------------------ waiting out an outage
@pytest.fixture
def results(make_transaction):
    return SCORER.score([make_transaction(f"tx-{i}", 99) for i in range(3)])


def test_a_working_store_records_at_once_without_pausing(store, results):
    consumer, stats = FakeConsumer(), ConsumeStats()
    assert asyncio.run(record_until_done(results, consumer, store, stats, None, FakeClock()))
    assert consumer.events == []
    assert (stats.outages, stats.seconds_waiting) == (0, 0)
    assert store.get("tx-0").decision == "block"


def test_an_outage_pauses_retries_with_growing_delays_and_resumes(store, results):
    clock = FakeClock()
    consumer, stats = FakeConsumer(clock=clock), ConsumeStats()
    flaky = FlakyStore(store, failures=3)
    assert asyncio.run(record_until_done(results, consumer, flaky, stats, None, clock, (1, 2, 4)))
    assert consumer.events == ["pause", "resume"]
    assert clock.now == 1 + 2 + 4  # waited between the four attempts
    assert (stats.outages, stats.seconds_waiting) == (1, 7)
    assert store.get("tx-2").decision == "block"


def test_the_last_delay_repeats_for_a_long_outage(store, results):
    clock = FakeClock()
    consumer, flaky = FakeConsumer(clock=clock), FlakyStore(store, failures=5)
    asyncio.run(record_until_done(results, consumer, flaky, ConsumeStats(), None, clock, (1, 2)))
    assert clock.now == 1 + 2 + 2 + 2 + 2


def test_retries_do_not_count_as_lost_decisions(store, results):
    clock = FakeClock()
    consumer, flaky = FakeConsumer(clock=clock), FlakyStore(store, failures=1)
    asyncio.run(record_until_done(results, consumer, flaky, ConsumeStats(), None, clock, (1,)))
    assert flaky.calls == [True, True]


def test_stopping_during_an_outage_gives_up_and_resumes_partitions(results):
    consumer = FakeConsumer()

    def attempt(stop):
        return record_until_done(
            results, consumer, FlakyStore(), ConsumeStats(), stop, consumer.clock, (0.01,)
        )

    assert asyncio.run(stop_soon(attempt)) is False
    assert consumer.events == ["pause", "resume"]


def test_messages_fetched_while_paused_are_put_back(make_transaction):
    clock = FakeClock()
    stray = {TP1: [message(make_transaction("tx-9"), offset=40, partition=1)]}
    consumer = FakeConsumer(clock=clock, strays=[stray])
    consumer.pause(TP0, TP1)
    asyncio.run(wait_polling(consumer, 2, None, clock))
    assert consumer.seeks == [(TP1, 40)]


# ------------------------------------------------------------ handling a batch
def batch_of(make_transaction):
    return {
        TP0: [
            message(make_transaction("tx-1", 99), offset=0),
            message(key=b"x", value=b"{", offset=1),
        ],
        TP1: [message(make_transaction("tx-2", 10), offset=3, partition=1)],
    }


def test_a_batch_is_dead_lettered_scored_recorded_and_then_committed(store, make_transaction):
    consumer, producer = FakeConsumer(), FakeProducer()
    stats = asyncio.run(
        handle_batch(batch_of(make_transaction), consumer, producer, "dlq", SCORER, store)
    )
    assert (stats.consumed, stats.scored, stats.dead_lettered) == (3, 2, 1)
    assert stats.decisions == {Decision.BLOCK: 1, Decision.APPROVE: 1}
    assert [topic for topic, *_ in producer.sent] == ["dlq"]
    assert store.get("tx-1").decision == "block"
    assert consumer.commits == [{TP0: 2, TP1: 4}]


def test_dead_letters_are_confirmed_before_the_commit(store, make_transaction):
    consumer = FakeConsumer()
    producer = FakeProducer(events=consumer.events)
    asyncio.run(handle_batch(batch_of(make_transaction), consumer, producer, "dlq", SCORER, store))
    assert consumer.events == ["dead-letter", "commit"]


def test_a_batch_recorded_after_an_outage_is_committed_after_it(store, make_transaction):
    clock = FakeClock()
    consumer = FakeConsumer(clock=clock)
    stats = asyncio.run(
        handle_batch(
            batch_of(make_transaction),
            consumer,
            FakeProducer(),
            "dlq",
            SCORER,
            FlakyStore(store, failures=2),
            clock=clock,
            retry_delays=(1,),
        )
    )
    assert consumer.events == ["pause", "resume", "commit"]
    assert stats.outages == 1
    assert store.get("tx-1").decision == "block"


def test_nothing_is_committed_when_stopped_during_an_outage(make_transaction):
    consumer = FakeConsumer()

    def attempt(stop):
        return handle_batch(
            batch_of(make_transaction),
            consumer,
            FakeProducer(),
            "dlq",
            SCORER,
            FlakyStore(),
            stop=stop,
            retry_delays=(0.01,),
        )

    stats = asyncio.run(stop_soon(attempt))
    assert stats.interrupted
    assert (stats.outages, stats.consumed, stats.batches) == (1, 0, 0)
    assert consumer.commits == []


def test_a_commit_lost_to_a_rebalance_is_logged_not_raised(store, make_transaction, caplog):
    consumer = FakeConsumer(commit_error=CommitFailedError("rebalanced"))
    with caplog.at_level(logging.WARNING, logger="src.streaming.consumer"):
        asyncio.run(
            handle_batch(batch_of(make_transaction), consumer, FakeProducer(), "dlq", SCORER, store)
        )
    assert "the batch will be re-read" in caplog.text


# ------------------------------------------------------------------ the loop
def test_the_loop_stops_once_idle_and_adds_up_every_batch(store, make_transaction):
    clock = FakeClock()
    batches = [
        {TP0: [message(make_transaction("tx-1"), offset=0)]},
        {TP0: [message(make_transaction("tx-2"), offset=1)]},
    ]
    consumer = FakeConsumer(batches, clock=clock)
    stats = asyncio.run(
        consume(consumer, FakeProducer(), SCORER, store, "dlq", 500, until_idle=3, clock=clock)
    )
    assert (stats.batches, stats.scored) == (2, 2)
    assert clock.now == 3
    assert len(consumer.commits) == 2


def test_the_loop_stops_when_asked(store):
    def attempt(stop):
        stop.set()
        return consume(FakeConsumer(), FakeProducer(), SCORER, store, "dlq", 500, stop)

    assert asyncio.run(stop_soon(attempt)).batches == 0


def test_the_loop_ends_when_stopped_during_an_outage(make_transaction):
    consumer = FakeConsumer([{TP0: [message(make_transaction("tx-1"), offset=0)]}])

    def attempt(stop):
        return consume(
            consumer,
            FakeProducer(),
            SCORER,
            FlakyStore(),
            "dlq",
            500,
            stop,
            retry_delays=(0.01,),
        )

    stats = asyncio.run(stop_soon(attempt))
    assert (stats.batches, stats.outages, consumer.commits) == (0, 1, [])


def test_the_summary_counts_everything():
    stats = ConsumeStats(
        consumed=12, scored=10, dead_lettered=1, duplicates=1, batches=2, seconds=2.0
    )
    stats.decisions.update({Decision.APPROVE: 8, Decision.REVIEW: 1, Decision.BLOCK: 1})
    assert stats.summary() == (
        "consumed 12 message(s) in 2 batch(es), 2.0s (6/s): scored 10 "
        "(8 approve, 1 review, 1 block), 1 dead-lettered, 1 duplicate(s) skipped"
    )
    stats.outages, stats.seconds_waiting = 1, 42.4
    assert stats.summary().endswith("; waited 42s for the database in 1 outage(s)")


# ---------------------------------------------------------------- the script
@pytest.mark.parametrize("args", [["--until-idle", "0"], ["--max-batch", "0"]])
def test_the_script_rejects_bad_arguments(args):
    with pytest.raises(SystemExit):
        consume_script.main(args)


def test_the_script_fails_cleanly_when_no_broker_answers(capsys):
    assert consume_script.main(["--bootstrap-servers", "127.0.0.1:1", "--until-idle", "1"]) == 1
    assert "Kafka error at 127.0.0.1:1" in capsys.readouterr().err


def test_the_consumer_module_logs_under_its_own_name():
    assert consumer_module.log.name == "src.streaming.consumer"


# ------------------------------------------------------------ through Kafka
async def send_raw(bootstrap, topic, records):
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        for key, value in records:
            await producer.send_and_wait(topic, value, key=key)
    finally:
        await producer.stop()


async def read_dead_letters(bootstrap, topic):
    consumer = AIOKafkaConsumer(
        topic, bootstrap_servers=bootstrap, group_id="dlq-check", auto_offset_reset="earliest"
    )
    await consumer.start()
    try:
        batches = await consumer.getmany(timeout_ms=5000)
        return [m for batch in batches.values() for m in batch]
    finally:
        await consumer.stop()


async def committed_total(bootstrap, topic, group, partitions=3):
    """The offsets `group` has committed on `topic`, summed over its partitions."""
    consumer = AIOKafkaConsumer(bootstrap_servers=bootstrap, group_id=group)
    await consumer.start()
    try:
        committed = [await consumer.committed(TopicPartition(topic, p)) for p in range(partitions)]
        return sum(offset or 0 for offset in committed)
    finally:
        await consumer.stop()


@pytest.fixture
def stream(kafka_bootstrap, make_topic, make_transaction):
    """A 3-partition topic holding 30 transactions, 5 re-sent copies and 2 broken messages."""
    topic, dlq = make_topic(partitions=3), make_topic()
    amounts = [10.0] * 26 + [50.0] * 2 + [99.0] * 2  # 26 approve, 2 review, 2 block
    transactions = [make_transaction(f"tx-{i}", a) for i, a in enumerate(amounts)]
    records = [
        (encode_transaction(t).key, encode_transaction(t).value)
        for t in transactions + transactions[:5]
    ]
    records += [(b"broken-1", b"not json"), (b"broken-2", b'{"transaction_id": "broken-2"}')]
    asyncio.run(send_raw(kafka_bootstrap, topic, records))
    return SimpleNamespace(topic=topic, dlq=dlq, bootstrap=kafka_bootstrap, sent=len(records))


def run_on(stream, store, group, stop=None, until_idle=2.0):
    return run(
        stop=stop,
        until_idle=until_idle,
        group=group,
        topic=stream.topic,
        dlq_topic=stream.dlq,
        bootstrap_servers=stream.bootstrap,
        scorer=SCORER,
        store=store,
        retry_delays=(0.2,),
    )


def consume_stream(stream, store, group, until_idle=2.0):
    return asyncio.run(run_on(stream, store, group, until_idle=until_idle))


@pytest.mark.kafka
def test_a_stream_is_scored_recorded_and_committed(stream, store):
    group = f"g-{uuid4().hex[:8]}"
    stats = consume_stream(stream, store, group)

    assert stats.consumed == stream.sent == 37
    assert stats.dead_lettered == 2
    # Re-sent copies are either dropped within a batch or re-recorded as upserts.
    assert stats.scored + stats.duplicates == 35
    assert len(store.recent(limit=100)) == 30
    counts = {d: len(store.recent(d, limit=100)) for d in Decision}
    assert counts == {Decision.APPROVE: 26, Decision.REVIEW: 2, Decision.BLOCK: 2}
    assert asyncio.run(committed_total(stream.bootstrap, stream.topic, group)) == stream.sent


@pytest.mark.kafka
def test_broken_messages_reach_the_dead_letter_topic_with_reasons(stream, store):
    consume_stream(stream, store, f"g-{uuid4().hex[:8]}")
    dead = asyncio.run(read_dead_letters(stream.bootstrap, stream.dlq))
    reasons = {m.key: header(m.headers, REASON_HEADER) for m in dead}
    assert "Invalid JSON" in reasons[b"broken-1"]
    assert "V1: Field required" in reasons[b"broken-2"]
    assert {m.value for m in dead} == {b"not json", b'{"transaction_id": "broken-2"}'}


@pytest.mark.kafka
def test_a_restarted_group_does_not_read_committed_messages_again(stream, store):
    group = f"g-{uuid4().hex[:8]}"
    consume_stream(stream, store, group)
    assert consume_stream(stream, store, group, until_idle=1.5).consumed == 0


@pytest.mark.kafka
def test_a_new_group_replays_everything_without_duplicating_rows(stream, store):
    consume_stream(stream, store, f"g-{uuid4().hex[:8]}")
    replay = consume_stream(stream, store, f"replay-{uuid4().hex[:8]}")
    assert replay.consumed == stream.sent
    assert len(store.recent(limit=100)) == 30


@pytest.mark.kafka
def test_an_outage_is_waited_out_and_no_decision_is_lost(stream, store):
    group = f"g-{uuid4().hex[:8]}"
    stats = consume_stream(stream, FlakyStore(store, failures=3), group)
    assert stats.outages >= 1
    assert stats.consumed == stream.sent
    assert len(store.recent(limit=100)) == 30
    assert asyncio.run(committed_total(stream.bootstrap, stream.topic, group)) == stream.sent


@pytest.mark.kafka
def test_stopping_during_an_outage_commits_nothing_and_a_restart_recovers(stream, store):
    group = f"g-{uuid4().hex[:8]}"
    stats = asyncio.run(
        stop_soon(lambda stop: run_on(stream, FlakyStore(), group, stop, None), after=3.0)
    )
    assert stats.outages == 1
    assert asyncio.run(committed_total(stream.bootstrap, stream.topic, group)) == 0

    recovered = consume_stream(stream, store, group)
    assert recovered.consumed == stream.sent
    assert len(store.recent(limit=100)) == 30
