"""Step 4.1 guard rails: the Kafka broker and its topics.

Tests marked `kafka` need the Compose broker (`make kafka`) and are skipped
without it. They write only to throwaway topics, never to the real ones.
"""

import asyncio

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from scripts import check_kafka
from src.config import settings
from src.streaming.kafka import MissingTopicsError, admin_client, check_topics


def consumer(topic, bootstrap, group):
    return AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )


async def send(bootstrap, topic, *values, key=None):
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        for value in values:
            await producer.send_and_wait(topic, value, key=key)
    finally:
        await producer.stop()


async def read(topic, bootstrap, group, commit=True):
    """Everything the group has not yet committed, then (optionally) commit it."""
    client = consumer(topic, bootstrap, group)
    await client.start()
    try:
        batches = await client.getmany(timeout_ms=5000)
        messages = [message for batch in batches.values() for message in batch]
        if commit:
            await client.commit()
        return messages
    finally:
        await client.stop()


# ------------------------------------------------------------- without a broker
def test_a_missing_topic_error_says_how_to_create_it():
    error = MissingTopicsError(["transactions.dlq", "transactions"])
    assert error.topics == ["transactions", "transactions.dlq"]
    assert "docker compose up -d kafka-init" in str(error)


def test_the_check_script_fails_when_no_broker_answers(capsys):
    # Port 1 on this machine: nothing listens there.
    assert check_kafka.main(["--bootstrap-servers", "127.0.0.1:1"]) == 1
    assert "no Kafka answering at 127.0.0.1:1" in capsys.readouterr().err


# ---------------------------------------------------------------- with a broker
@pytest.mark.kafka
def test_the_configured_topics_exist_with_their_partitions(kafka_bootstrap):
    counts = asyncio.run(check_topics(bootstrap_servers=kafka_bootstrap))
    assert counts == {settings.kafka_topic: 3, settings.kafka_dlq_topic: 1}


@pytest.mark.kafka
def test_a_missing_topic_is_named_and_not_created(kafka_bootstrap):
    with pytest.raises(MissingTopicsError) as raised:
        asyncio.run(check_topics(["no-such-topic"], kafka_bootstrap))
    assert raised.value.topics == ["no-such-topic"]

    async def topic_names():
        async with admin_client(kafka_bootstrap) as admin:
            return await admin.list_topics()

    assert "no-such-topic" not in asyncio.run(topic_names())


@pytest.mark.kafka
def test_a_keyed_message_round_trips(kafka_bootstrap, throwaway_topic):
    asyncio.run(send(kafka_bootstrap, throwaway_topic, b'{"Amount": 12.5}', key=b"tx-1"))
    (message,) = asyncio.run(read(throwaway_topic, kafka_bootstrap, "round-trip"))
    assert (message.key, message.value, message.offset) == (b"tx-1", b'{"Amount": 12.5}', 0)


@pytest.mark.kafka
def test_a_consumer_group_resumes_after_its_last_committed_message(
    kafka_bootstrap, throwaway_topic
):
    """The bookmark the Phase 4 consumer relies on: committed messages are not re-read."""
    asyncio.run(send(kafka_bootstrap, throwaway_topic, b"first"))
    assert [m.value for m in asyncio.run(read(throwaway_topic, kafka_bootstrap, "g"))] == [b"first"]

    asyncio.run(send(kafka_bootstrap, throwaway_topic, b"second"))
    assert [m.value for m in asyncio.run(read(throwaway_topic, kafka_bootstrap, "g"))] == [
        b"second"
    ]


@pytest.mark.kafka
def test_uncommitted_messages_are_delivered_again(kafka_bootstrap, throwaway_topic):
    """At-least-once: a consumer that stops before committing sees the messages again."""
    asyncio.run(send(kafka_bootstrap, throwaway_topic, b"first", b"second"))
    first_try = asyncio.run(read(throwaway_topic, kafka_bootstrap, "g", commit=False))
    second_try = asyncio.run(read(throwaway_topic, kafka_bootstrap, "g"))
    assert [m.value for m in first_try] == [m.value for m in second_try] == [b"first", b"second"]


@pytest.mark.kafka
def test_a_new_group_reads_the_whole_history(kafka_bootstrap, throwaway_topic):
    """Why a new model can be tested by replaying the topic under a new group name."""
    asyncio.run(send(kafka_bootstrap, throwaway_topic, b"first", b"second"))
    asyncio.run(read(throwaway_topic, kafka_bootstrap, "live-model"))
    replay = asyncio.run(read(throwaway_topic, kafka_bootstrap, "candidate-model"))
    assert [m.value for m in replay] == [b"first", b"second"]


@pytest.mark.kafka
def test_the_check_script_reports_the_topics(kafka_bootstrap, capsys):
    assert check_kafka.main(["--bootstrap-servers", kafka_bootstrap]) == 0
    out = capsys.readouterr().out
    assert f"{settings.kafka_topic} (3 partitions)" in out
    assert f"{settings.kafka_dlq_topic} (1 partition)" in out
