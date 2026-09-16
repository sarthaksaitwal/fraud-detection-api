"""Step 4.3 guard rails: the producer."""

import asyncio
from types import SimpleNamespace

import pytest
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.errors import KafkaError

from scripts import produce as produce_script
from src.api.schemas import Transaction
from src.ml.preprocess import RAW_FEATURES
from src.streaming.kafka import MissingTopicsError
from src.streaming.messages import decode_transaction
from src.streaming.producer import ProduceReport, load_transactions, produce, run


class FakeTime:
    """A clock that only moves when the code under test sleeps or a send takes time."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProducer:
    """Records what is sent. Each send can cost time, fail, or run a callback."""

    def __init__(self, time=None, send_seconds=0.0, fail_at=None, on_send=None):
        self.time, self.send_seconds, self.fail_at, self.on_send = (
            time,
            send_seconds,
            fail_at,
            on_send,
        )
        self.sent = []

    async def send(self, topic, value, key=None):
        if self.time is not None:
            self.time.now += self.send_seconds
        future = asyncio.get_running_loop().create_future()
        if len(self.sent) == self.fail_at:
            future.set_exception(KafkaError("broker went away"))
        else:
            future.set_result(SimpleNamespace(partition=len(self.sent) % 3))
        self.sent.append((topic, key, value))
        if self.on_send:
            self.on_send(len(self.sent))
        return future


@pytest.fixture
def transactions(raw_df):
    rows = raw_df[RAW_FEATURES].head(5).to_dict("records")
    return [Transaction(transaction_id=f"tx-{i}", **row) for i, row in enumerate(rows)]


def send(transactions, producer, rate=0.0, stop=None, time=None, start=0):
    time = time or FakeTime()
    report = ProduceReport(start=start)
    asyncio.run(produce(transactions, producer, "t", rate, report, stop, time.clock, time.sleep))
    return report


# ------------------------------------------------------------------ sending
def test_every_transaction_is_sent_in_order_keyed_by_its_id(transactions):
    producer = FakeProducer()
    report = send(transactions, producer)
    assert [key for _, key, _ in producer.sent] == [b"tx-0", b"tx-1", b"tx-2", b"tx-3", b"tx-4"]
    assert [decode_transaction(k, v) for _, k, v in producer.sent] == transactions
    assert report.sent == 5
    assert report.partitions == {0: 2, 1: 2, 2: 1}


def test_messages_follow_the_rate(transactions):
    time = FakeTime()
    send(transactions, FakeProducer(time), rate=10, time=time)
    # Five messages at 10/s: the last is due 0.4s after the first.
    assert time.now == pytest.approx(0.4)


def test_the_time_a_send_takes_does_not_slow_the_stream(transactions):
    """Pacing by schedule, not by sleeping 1/rate after each send."""
    time = FakeTime()
    send(transactions, FakeProducer(time, send_seconds=0.05), rate=10, time=time)
    # Sleeping 0.1s after each 0.05s send would take 0.75s; the schedule takes 0.45s.
    assert time.now == pytest.approx(0.45)


def test_a_producer_behind_schedule_does_not_sleep(transactions):
    time = FakeTime()
    send(transactions, FakeProducer(time, send_seconds=0.2), rate=10, time=time)
    assert time.sleeps == []


def test_rate_zero_never_sleeps(transactions):
    time = FakeTime()
    send(transactions, FakeProducer(time), rate=0, time=time)
    assert time.sleeps == []


def test_stopping_sends_nothing_more_and_says_where_to_resume(transactions):
    async def scenario():
        stop = asyncio.Event()
        producer = FakeProducer(on_send=lambda count: count == 3 and stop.set())
        report = ProduceReport(start=100)
        await produce(transactions, producer, "t", 0, report, stop)
        return producer, report

    producer, report = asyncio.run(scenario())
    assert len(producer.sent) == 3
    assert (report.sent, report.next_start) == (3, 103)


def test_a_failed_send_stops_the_run_and_counts_only_confirmed_messages(transactions):
    producer = FakeProducer(fail_at=2)
    report = ProduceReport(start=0)
    with pytest.raises(KafkaError):
        asyncio.run(produce(transactions, producer, "t", 0, report))
    assert report.sent == 2
    assert report.next_start == 2


def test_the_summary_says_what_was_sent_and_how_to_resume():
    report = ProduceReport(start=10, sent=40, seconds=2.0)
    report.partitions.update({0: 14, 1: 13, 2: 13})
    assert report.summary("transactions") == (
        "sent 40 transaction(s) to transactions in 2.0s (20.0/s; p0: 14, p1: 13, p2: 13); "
        "resume with --start 50"
    )


# ------------------------------------------------------------ the source rows
@pytest.fixture
def test_split(raw_df, tmp_path):
    path = tmp_path / "test.parquet"
    raw_df.to_parquet(path)
    return path, raw_df


def test_transactions_come_in_time_order_with_row_ids(test_split):
    path, raw_df = test_split
    loaded = load_transactions(path=path)
    assert len(loaded) == len(raw_df)
    assert [t.Time for t in loaded] == sorted(raw_df["Time"])
    first = raw_df["Time"].idxmin()
    assert loaded[0].transaction_id == f"test-row-{first}"
    assert loaded[0].features() == raw_df.loc[first, RAW_FEATURES].to_dict()


def test_start_and_limit_select_a_window_of_the_same_order(test_split):
    path, _ = test_split
    everything = load_transactions(path=path)
    assert load_transactions(start=10, limit=5, path=path) == everything[10:15]
    assert load_transactions(start=1990, path=path) == everything[1990:]


def test_equal_times_keep_a_stable_order(test_split, tmp_path):
    _, raw_df = test_split
    tied = raw_df.head(20).assign(Time=5.0)
    path = tmp_path / "tied.parquet"
    tied.to_parquet(path)
    assert [t.transaction_id for t in load_transactions(path=path)] == [
        f"test-row-{row}" for row in sorted(tied.index)
    ]


def test_a_start_past_the_end_is_an_error(test_split):
    path, raw_df = test_split
    with pytest.raises(ValueError, match="past the end"):
        load_transactions(start=len(raw_df), path=path)


# ---------------------------------------------------------------- the script
@pytest.mark.parametrize("args", [["--rate", "-1"], ["--limit", "0"], ["--start", "-5"]])
def test_the_script_rejects_bad_arguments(args):
    with pytest.raises(SystemExit):
        produce_script.main(args)


def test_the_script_fails_cleanly_when_no_broker_answers(transactions, monkeypatch, capsys):
    monkeypatch.setattr(produce_script, "load_transactions", lambda start, limit: transactions)
    assert produce_script.main(["--bootstrap-servers", "127.0.0.1:1"]) == 1
    assert "Kafka error at 127.0.0.1:1" in capsys.readouterr().err


# ------------------------------------------------------------ through Kafka
@pytest.fixture
def three_partition_topic(make_topic):
    return make_topic(partitions=3)


async def read_all(bootstrap, topic, expected):
    consumer = AIOKafkaConsumer(
        topic, bootstrap_servers=bootstrap, group_id="check", auto_offset_reset="earliest"
    )
    await consumer.start()
    try:
        messages = []
        while len(messages) < expected:
            batches = await consumer.getmany(timeout_ms=5000)
            if not batches:
                break
            messages += [message for batch in batches.values() for message in batch]
        partitions = [TopicPartition(topic, p) for p in consumer.partitions_for_topic(topic)]
        end_offsets = await consumer.end_offsets(partitions)
        return messages, sum(end_offsets.values())
    finally:
        await consumer.stop()


@pytest.fixture
def many(raw_df):
    rows = raw_df[RAW_FEATURES].head(30).to_dict("records")
    return [Transaction(transaction_id=f"tx-{i}", **row) for i, row in enumerate(rows)]


@pytest.mark.kafka
def test_every_confirmed_transaction_is_in_the_topic(kafka_bootstrap, three_partition_topic, many):
    report = asyncio.run(
        run(
            rate=0,
            topic=three_partition_topic,
            bootstrap_servers=kafka_bootstrap,
            transactions=many,
        )
    )
    messages, stored = asyncio.run(read_all(kafka_bootstrap, three_partition_topic, 30))
    assert report.sent == stored == 30
    assert sorted(report.partitions) == [0, 1, 2]
    assert {decode_transaction(m.key, m.value).transaction_id for m in messages} == {
        t.transaction_id for t in many
    }


@pytest.mark.kafka
def test_a_resent_transaction_lands_on_the_same_partition(
    kafka_bootstrap, three_partition_topic, many
):
    for _ in range(2):
        asyncio.run(
            run(
                rate=0,
                topic=three_partition_topic,
                bootstrap_servers=kafka_bootstrap,
                transactions=many,
            )
        )
    messages, _ = asyncio.run(read_all(kafka_bootstrap, three_partition_topic, 60))
    partitions_by_key = {}
    for message in messages:
        partitions_by_key.setdefault(message.key, set()).add(message.partition)
    assert len(partitions_by_key) == 30
    assert all(len(partitions) == 1 for partitions in partitions_by_key.values())


@pytest.mark.kafka
def test_a_real_run_follows_the_rate(kafka_bootstrap, throwaway_topic, many):
    report = asyncio.run(
        run(
            rate=100,
            topic=throwaway_topic,
            bootstrap_servers=kafka_bootstrap,
            transactions=many[:20],
        )
    )
    # 20 messages at 100/s: the last is due 0.19s after the first.
    assert 0.18 <= report.seconds < 1.0


@pytest.mark.kafka
def test_sending_to_a_missing_topic_fails_before_connecting_a_producer(kafka_bootstrap, many):
    with pytest.raises(MissingTopicsError):
        asyncio.run(
            run(topic="no-such-topic", bootstrap_servers=kafka_bootstrap, transactions=many)
        )


@pytest.mark.kafka
def test_the_script_sends_and_reports(kafka_bootstrap, throwaway_topic, many, monkeypatch, capsys):
    monkeypatch.setattr(produce_script, "load_transactions", lambda start, limit: many[:5])
    args = ["--rate", "0", "--topic", throwaway_topic, "--bootstrap-servers", kafka_bootstrap]
    assert produce_script.main(args) == 0
    assert f"sent 5 transaction(s) to {throwaway_topic}" in capsys.readouterr().out
