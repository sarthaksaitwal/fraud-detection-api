"""Connecting to Kafka and checking its topics (Step 4.1).

Topics are created by the kafka-init service in docker-compose.yml, and the
broker does not create them on demand. A consumer subscribed to a topic that
does not exist does not fail: it waits, silently, forever. So producers and
consumers check their topics first and refuse to start without them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager

from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.errors import KafkaError

from src.config import settings

# Kafka's error code for a topic that does not exist.
UNKNOWN_TOPIC_OR_PARTITION = 3


class MissingTopicsError(RuntimeError):
    def __init__(self, topics: Iterable[str]) -> None:
        self.topics = sorted(topics)
        super().__init__(
            f"Kafka topics not found: {', '.join(self.topics)}. They are created by the "
            "kafka-init service: run `docker compose up -d kafka-init`."
        )


def configured_topics() -> list[str]:
    """Every topic this project reads or writes."""
    return [settings.kafka_topic, settings.kafka_dlq_topic]


@asynccontextmanager
async def admin_client(bootstrap_servers: str | None = None) -> AsyncIterator[AIOKafkaAdminClient]:
    """A connected admin client. Raises KafkaConnectionError if no broker answers."""
    admin = AIOKafkaAdminClient(
        bootstrap_servers=bootstrap_servers or settings.kafka_bootstrap_servers
    )
    try:
        await admin.start()
        yield admin
    finally:
        await admin.close()


async def partition_counts(admin: AIOKafkaAdminClient, topics: Iterable[str]) -> dict[str, int]:
    """Partitions per topic. Raises MissingTopicsError naming every topic that is absent."""
    counts, missing = {}, []
    for description in await admin.describe_topics(list(topics)):
        if description["error_code"] == UNKNOWN_TOPIC_OR_PARTITION:
            missing.append(description["topic"])
        elif description["error_code"]:
            raise KafkaError(
                f"describing topic {description['topic']!r} failed with Kafka error "
                f"code {description['error_code']}"
            )
        else:
            counts[description["topic"]] = len(description["partitions"])
    if missing:
        raise MissingTopicsError(missing)
    return counts


async def check_topics(
    topics: Iterable[str] | None = None, bootstrap_servers: str | None = None
) -> dict[str, int]:
    """Connect, and confirm the topics exist. Returns partitions per topic."""
    async with admin_client(bootstrap_servers) as admin:
        return await partition_counts(admin, configured_topics() if topics is None else topics)
