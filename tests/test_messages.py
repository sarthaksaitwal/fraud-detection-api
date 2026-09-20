"""Step 4.2 guard rails: transactions as Kafka messages."""

import asyncio
import json
import math
from datetime import datetime, timezone

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.api.schemas import Transaction
from src.features.entities import entities_for
from src.ml.preprocess import RAW_FEATURES
from src.streaming.messages import (
    FAILED_AT_HEADER,
    MAX_MESSAGE_BYTES,
    REASON_HEADER,
    SOURCE_OFFSET_HEADER,
    SOURCE_PARTITION_HEADER,
    SOURCE_TOPIC_HEADER,
    InvalidMessageError,
    dead_letter,
    decode_transaction,
    encode_transaction,
    header,
)


@pytest.fixture
def transaction(raw_df):
    return Transaction(transaction_id="tx-1", **raw_df[RAW_FEATURES].iloc[0].to_dict())


@pytest.fixture
def body(transaction):
    """The message value as a dict, to break in different ways."""
    return json.loads(encode_transaction(transaction).value)


def raw(body):
    return json.dumps(body).encode()


def reason_for(key, value):
    with pytest.raises(InvalidMessageError) as raised:
        decode_transaction(key, value)
    return raised.value.reason


# ------------------------------------------------------------------ encoding
def test_a_transaction_round_trips_exactly(transaction):
    record = encode_transaction(transaction)
    decoded = decode_transaction(record.key, record.value)
    assert decoded == transaction
    # Exact float equality: the model scores the same numbers whichever way they came.
    assert decoded.features() == transaction.features()


def test_the_key_is_the_transaction_id(transaction):
    assert encode_transaction(transaction).key == b"tx-1"


def test_the_value_is_the_body_post_score_accepts(transaction):
    value = json.loads(encode_transaction(transaction).value)
    assert list(value) == ["transaction_id", *RAW_FEATURES, "card_id", "merchant", "country"]
    assert Transaction(**value) == transaction


def test_the_card_merchant_and_country_survive_the_round_trip(raw_df):
    """Step 5.2: what velocity counts per has to reach the consumer."""
    transaction = Transaction(
        transaction_id="tx-1",
        **entities_for("tx-1"),
        **raw_df[RAW_FEATURES].iloc[0].to_dict(),
    )
    record = encode_transaction(transaction)
    decoded = decode_transaction(record.key, record.value)
    assert (decoded.card_id, decoded.merchant, decoded.country) == (
        transaction.card_id,
        transaction.merchant,
        transaction.country,
    )


def test_a_message_without_a_key_is_accepted(transaction):
    assert decode_transaction(None, encode_transaction(transaction).value) == transaction


# ---------------------------------------------------------------- rejections
def test_a_missing_transaction_id_is_rejected_on_the_stream(body):
    """Over HTTP it would be generated; here a redelivery would get a different id."""
    del body["transaction_id"]
    assert reason_for(None, raw(body)) == "transaction_id: required on the stream"


def test_a_key_that_disagrees_with_the_body_is_rejected(transaction):
    record = encode_transaction(transaction)
    assert reason_for(b"tx-2", record.value) == "key does not match transaction_id"


@pytest.mark.parametrize(
    "value, reason",
    [
        (None, "empty message"),
        (b"", "empty message"),
        (b"hello", "Invalid JSON"),
        (b"[1, 2, 3]", "Input should be an object"),
        (b'{"transaction_id": "\xff"}', "Invalid JSON"),
    ],
)
def test_a_message_that_is_not_a_json_object_is_rejected(value, reason):
    assert reason in reason_for(None, value)


def test_a_missing_field_is_named(body):
    del body["V14"]
    assert reason_for(None, raw(body)) == "V14: Field required"


def test_an_unknown_field_is_named(body):
    body["Class"] = 1
    assert reason_for(None, raw(body)) == "Class: Extra inputs are not permitted"


def test_nan_is_rejected(body):
    body["V3"] = math.nan
    assert reason_for(None, raw(body)) == "V3: Input should be a finite number"


def test_every_problem_is_listed(body):
    del body["V14"]
    body["Amount"] = -5
    reason = reason_for(None, raw(body))
    assert "V14: Field required" in reason
    assert "Amount: Input should be greater than or equal to 0" in reason


def test_an_oversized_message_is_rejected_before_parsing():
    reason = reason_for(None, b" " * (MAX_MESSAGE_BYTES + 1))
    assert reason.startswith(f"message is {MAX_MESSAGE_BYTES + 1} bytes")


def test_a_reason_never_contains_the_rejected_values(body):
    body["Amount"] = -123.456789
    body["V1"] = "secret-looking text"
    reason = reason_for(None, raw(body))
    assert "123.456789" not in reason
    assert "secret-looking text" not in reason


# -------------------------------------------------------------- dead letters
def test_a_dead_letter_keeps_the_original_bytes_and_says_why_and_where():
    failed_at = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    record = dead_letter(b"tx-1", b"not json", "Invalid JSON", "transactions", 2, 4711, failed_at)
    assert (record.key, record.value) == (b"tx-1", b"not json")
    assert header(record.headers, REASON_HEADER) == "Invalid JSON"
    assert header(record.headers, SOURCE_TOPIC_HEADER) == "transactions"
    assert header(record.headers, SOURCE_PARTITION_HEADER) == "2"
    assert header(record.headers, SOURCE_OFFSET_HEADER) == "4711"
    assert header(record.headers, FAILED_AT_HEADER) == "2026-09-16T12:00:00+00:00"


def test_a_dead_letter_can_hold_an_empty_message():
    record = dead_letter(None, None, "empty message", "transactions", 0, 0)
    assert (record.key, record.value) == (None, b"")


def test_a_missing_header_reads_as_none():
    assert header((), REASON_HEADER) is None


# ------------------------------------------------------------ through Kafka
async def send_and_read(bootstrap, topic, record):
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(
            topic, record.value, key=record.key, headers=list(record.headers)
        )
    finally:
        await producer.stop()

    consumer = AIOKafkaConsumer(
        topic, bootstrap_servers=bootstrap, group_id="messages", auto_offset_reset="earliest"
    )
    await consumer.start()
    try:
        batches = await consumer.getmany(timeout_ms=5000)
        (message,) = [message for batch in batches.values() for message in batch]
        return message
    finally:
        await consumer.stop()


@pytest.mark.kafka
def test_an_encoded_transaction_survives_kafka(kafka_bootstrap, throwaway_topic, transaction):
    record = encode_transaction(transaction)
    message = asyncio.run(send_and_read(kafka_bootstrap, throwaway_topic, record))
    assert decode_transaction(message.key, message.value) == transaction


@pytest.mark.kafka
def test_dead_letter_headers_survive_kafka(kafka_bootstrap, throwaway_topic):
    record = dead_letter(b"tx-1", b"hello", "Invalid JSON", "transactions", 1, 99)
    message = asyncio.run(send_and_read(kafka_bootstrap, throwaway_topic, record))
    assert message.value == b"hello"
    assert header(message.headers, REASON_HEADER) == "Invalid JSON"
    assert header(message.headers, SOURCE_OFFSET_HEADER) == "99"
