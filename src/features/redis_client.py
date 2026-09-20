"""Connecting to Redis, which holds what each card did recently (Step 5.1).

Velocity features ("how many transactions has this card made in the last
minute?") are read and written in front of every score, inside the API's
latency budget. Postgres already holds every decision for good, but answering
that question from a growing table, twice per transaction, is the wrong shape
of work. Redis keeps only the recent past, in memory, and expires it by itself.

Nothing stored here is worth keeping. It is derived from transactions Kafka
still has, and rebuilds itself as soon as they flow again, so the container
runs with persistence switched off. That is also why losing Redis must never
cost a decision: the service scores without the features instead (Step 5.6).

The client is synchronous, like the decision store. The consumer is async and
calls both from a worker thread, so there is one implementation rather than a
sync and an async copy of every velocity query.
"""

from __future__ import annotations

import logging

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from src.config import settings

log = logging.getLogger("src.features")

# A connection is checked with PING if it has been idle this long, so a server
# that restarted is noticed when the connection is taken from the pool rather
# than in the middle of a transaction's velocity lookup.
HEALTH_CHECK_INTERVAL_SECONDS = 30


def create_redis(url: str | None = None, timeout: float | None = None) -> Redis:
    """A Redis connection pool that fails fast. Defaults to REDIS_URL.

    redis-py retries a failed command ten times by default, with a growing
    delay between attempts. That is the opposite of what scoring needs: with
    Redis down, every transaction would wait through all ten before being
    scored without its features. One attempt, one short timeout, move on.
    """
    timeout = settings.redis_timeout_seconds if timeout is None else timeout
    return Redis.from_url(
        url or settings.redis_url,
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
        retry=Retry(NoBackoff(), retries=0),
        health_check_interval=HEALTH_CHECK_INTERVAL_SECONDS,
        # Keys, card ids and transaction ids are text, not bytes.
        decode_responses=True,
    )


def ping(client: Redis | None) -> bool:
    """Can Redis be reached? Never raises: the caller only wants yes or no."""
    if client is None:
        return False
    try:
        return bool(client.ping())
    except RedisError:
        return False


def open_configured_redis() -> Redis | None:
    """The client for REDIS_URL, or None when VELOCITY_FEATURES is false.

    A malformed URL is a configuration mistake and raises, so the service
    refuses to start. A Redis that is merely down is a warning: connections are
    opened lazily, and the first command after it comes back succeeds.
    """
    if not settings.velocity_features:
        log.info("velocity features are off: VELOCITY_FEATURES is false")
        return None
    client = create_redis()
    if not ping(client):
        log.warning(
            "redis unavailable at startup (%s); scoring continues without velocity features",
            settings.redis_url,
        )
    return client
