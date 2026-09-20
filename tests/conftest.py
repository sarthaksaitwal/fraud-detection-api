"""Fixtures shared across test modules."""

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from aiokafka.admin import NewTopic
from aiokafka.errors import KafkaError
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config import settings
from src.features.redis_client import create_redis
from src.ml.preprocess import TARGET, V_COLUMNS
from src.storage.db import create_db_engine
from src.storage.store import failure_reason
from src.streaming.kafka import MissingTopicsError, admin_client, partition_counts

# The columns that separate fraud most strongly in the real data (Step 1.1, Cell 9).
FRAUD_SIGNAL_COLUMNS = ["V3", "V10", "V12", "V14", "V17"]

# Redis database for tests, so they never touch database 0's velocity keys.
TEST_REDIS_DB = 15


@pytest.fixture(autouse=True)
def no_configured_database(monkeypatch):
    """No test writes to the database in .env. Tests that need one open their own."""
    monkeypatch.setattr(settings, "persist_decisions", False)


@pytest.fixture(autouse=True)
def no_configured_redis(monkeypatch):
    """No test touches the Redis in .env. Tests that need one open their own."""
    monkeypatch.setattr(settings, "velocity_features", False)


@pytest.fixture(scope="session")
def postgres_engine():
    """An engine on a throwaway schema in a real Postgres, or a skip if none answers.

    Uses TEST_DATABASE_URL, else DATABASE_URL: normally the Compose database
    from `make up`. Tests get their own schema, dropped afterwards, so they
    never see or touch the decisions already stored there.
    """
    url = os.environ.get("TEST_DATABASE_URL", settings.database_url)
    if not url.startswith("postgresql"):
        pytest.skip(f"not a Postgres URL: {url.split(':', 1)[0]}")

    schema = f"test_{uuid4().hex[:12]}"
    fast_fail = {"connect_timeout": 2}
    admin = create_db_engine(url, connect_args=fast_fail)
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    except SQLAlchemyError as exc:
        admin.dispose()
        pytest.skip(f"no Postgres answering at {admin.url!r}: {failure_reason(exc)}")

    # Every connection from this engine looks in the throwaway schema first.
    engine = create_db_engine(url, connect_args=fast_fail | {"options": f"-csearch_path={schema}"})
    yield engine
    engine.dispose()
    with admin.begin() as connection:
        connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    admin.dispose()


@pytest.fixture(scope="session")
def kafka_bootstrap():
    """The bootstrap servers of a running Kafka broker, or a skip if none answers.

    Uses TEST_KAFKA_BOOTSTRAP_SERVERS, else KAFKA_BOOTSTRAP_SERVERS: normally the
    Compose broker from `make kafka`.
    """
    servers = os.environ.get("TEST_KAFKA_BOOTSTRAP_SERVERS", settings.kafka_bootstrap_servers)

    async def connect():
        async with admin_client(servers):
            pass

    try:
        asyncio.run(connect())
    except KafkaError as exc:
        pytest.skip(f"no Kafka answering at {servers}: {type(exc).__name__}")
    return servers


@pytest.fixture
def make_topic(kafka_bootstrap):
    """Creates new topics for one test and deletes them afterwards.

    Tests never touch the real topics. Kafka confirms a new topic before it
    shows up in metadata, and code that checks for its topic straight away
    would call it missing, so this waits until the topic can be seen.
    """
    created = []

    async def create(name, partitions):
        async with admin_client(kafka_bootstrap) as admin:
            await admin.create_topics(
                [NewTopic(name, num_partitions=partitions, replication_factor=1)]
            )
            for _ in range(50):
                try:
                    if (await partition_counts(admin, [name]))[name] == partitions:
                        return
                except MissingTopicsError:
                    pass
                await asyncio.sleep(0.1)
            raise TimeoutError(f"topic {name} was created but never appeared")

    def make(partitions=1):
        name = f"test-{uuid4().hex[:12]}"
        asyncio.run(create(name, partitions))
        created.append(name)
        return name

    yield make

    async def delete():
        async with admin_client(kafka_bootstrap) as admin:
            await admin.delete_topics(created)

    if created:
        asyncio.run(delete())


@pytest.fixture
def throwaway_topic(make_topic):
    """A new one-partition topic, deleted after the test."""
    return make_topic()


@pytest.fixture(scope="session")
def redis_url():
    """A Redis URL on a database of its own, or a skip if no server answers.

    Uses TEST_REDIS_URL, else REDIS_URL with logical database 15 instead of 0:
    tests empty the database they use, and must never empty the one the service
    is filling with velocity keys.
    """
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        url = urlunsplit(urlsplit(settings.redis_url)._replace(path=f"/{TEST_REDIS_DB}"))

    client = create_redis(url)
    try:
        client.ping()
    except RedisError as exc:
        pytest.skip(f"no Redis answering at {url}: {type(exc).__name__}")
    finally:
        client.close()
    return url


@pytest.fixture
def redis_client(redis_url):
    """An empty Redis database for one test, emptied again afterwards."""
    client = create_redis(redis_url)
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """2,000 fake transactions shaped like the Kaggle data, 2.5% fraud.

    Fraud rows are shifted by -7 on the columns that separate fraud in the real
    data, so an anomaly detector has a real signal to find.
    """
    rng = np.random.default_rng(0)
    n, n_fraud = 2000, 50
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172_800, n))
    df["Amount"] = rng.exponential(80, n)
    df[TARGET] = 0
    df.loc[: n_fraud - 1, TARGET] = 1
    df.loc[: n_fraud - 1, FRAUD_SIGNAL_COLUMNS] -= 7
    return df
