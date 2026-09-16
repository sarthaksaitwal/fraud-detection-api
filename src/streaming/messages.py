"""The shape of a transaction on the Kafka topic (Step 4.2).

    key:    transaction_id, UTF-8
    value:  the transaction as a compact JSON object, the same body POST /score takes

The value is checked with the API's own Transaction schema, so the stream and
the HTTP endpoint accept exactly the same transactions. One rule is stricter:
`transaction_id` is required. Over HTTP a missing id is generated, which is
harmless. On the stream Kafka may deliver a message twice, and a generated id
would give each delivery a different id, so the database upsert would store the
same transaction twice instead of once.

A message that fails any check becomes an InvalidMessageError with a short
reason. The consumer sends it to the dead-letter topic with that reason and
carries on, rather than crashing on it again on every restart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from src.api.schemas import Transaction

# A transaction is about 1 KB of JSON. Anything much larger is not one of ours.
MAX_MESSAGE_BYTES = 16_384

# Header names on a dead-letter record. Kafka headers are (str, bytes) pairs.
REASON_HEADER = "dlq-reason"
SOURCE_TOPIC_HEADER = "dlq-source-topic"
SOURCE_PARTITION_HEADER = "dlq-source-partition"
SOURCE_OFFSET_HEADER = "dlq-source-offset"
FAILED_AT_HEADER = "dlq-failed-at"


class InvalidMessageError(ValueError):
    """A message that is not a valid transaction. `reason` never contains its values."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class KafkaRecord:
    """What a producer sends: a key, a value and optional headers."""

    key: bytes | None
    value: bytes
    headers: tuple[tuple[str, bytes], ...] = ()


def encode_transaction(transaction: Transaction) -> KafkaRecord:
    """The transaction as a keyed Kafka record.

    Keyed by transaction_id, so Kafka always routes a given transaction to the
    same partition: a re-sent transaction lands behind its first copy, never
    beside it in another partition.
    """
    return KafkaRecord(
        key=transaction.transaction_id.encode("utf-8"),
        value=transaction.model_dump_json().encode("utf-8"),
    )


def describe_validation_error(error: ValidationError) -> str:
    """Where and why validation failed, never the value.

    For example `V14: Field required; Amount: Input should be greater than or
    equal to 0`. Like the API's 422 handler, it leaves out the rejected input:
    the reason is stored in a Kafka header and printed in logs.
    """
    parts = []
    for detail in error.errors():
        where = ".".join(str(part) for part in detail["loc"])
        parts.append(f"{where}: {detail['msg']}" if where else detail["msg"])
    return "; ".join(parts)


def decode_transaction(key: bytes | None, value: bytes | None) -> Transaction:
    """The transaction in a Kafka record, or InvalidMessageError saying what is wrong."""
    if not value:
        raise InvalidMessageError("empty message")
    if len(value) > MAX_MESSAGE_BYTES:
        raise InvalidMessageError(
            f"message is {len(value)} bytes; a transaction is at most {MAX_MESSAGE_BYTES}"
        )
    try:
        transaction = Transaction.model_validate_json(value)
    except ValidationError as error:
        raise InvalidMessageError(describe_validation_error(error)) from None

    # model_fields_set lists the fields the message really contained, as opposed
    # to ones Pydantic filled in with a default.
    if "transaction_id" not in transaction.model_fields_set:
        raise InvalidMessageError("transaction_id: required on the stream")
    if key is not None and key != transaction.transaction_id.encode("utf-8"):
        raise InvalidMessageError("key does not match transaction_id")
    return transaction


def dead_letter(
    key: bytes | None,
    value: bytes | None,
    reason: str,
    topic: str,
    partition: int,
    offset: int,
    failed_at: datetime | None = None,
) -> KafkaRecord:
    """A dead-letter record: the original bytes, untouched, plus why and where they failed.

    The value is kept byte for byte, so once the cause is fixed the record can be
    sent back to the main topic exactly as it first arrived.
    """
    failed_at = failed_at or datetime.now(timezone.utc)
    headers = (
        (REASON_HEADER, reason.encode("utf-8")),
        (SOURCE_TOPIC_HEADER, topic.encode("utf-8")),
        (SOURCE_PARTITION_HEADER, str(partition).encode("ascii")),
        (SOURCE_OFFSET_HEADER, str(offset).encode("ascii")),
        (FAILED_AT_HEADER, failed_at.isoformat().encode("ascii")),
    )
    return KafkaRecord(key=key, value=value or b"", headers=headers)


def header(headers: Sequence[tuple[str, bytes]], name: str) -> str | None:
    """One header's value as text, or None if the record does not have it."""
    for header_name, header_value in headers:
        if header_name == name:
            return header_value.decode("utf-8")
    return None
