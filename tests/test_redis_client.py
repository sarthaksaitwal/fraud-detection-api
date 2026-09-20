"""Step 5.1: the Redis connection, and what happens when Redis is not there."""

import time

import pytest

from src.config import settings
from src.features.redis_client import create_redis, open_configured_redis, ping

# A port nothing listens on, so connecting fails immediately rather than hanging.
UNUSED_PORT_URL = "redis://127.0.0.1:6399/0"


def test_create_redis_uses_the_configured_url(monkeypatch):
    monkeypatch.setattr(settings, "redis_url", "redis://example.test:6380/3")
    client = create_redis()
    pool = client.connection_pool.connection_kwargs
    assert (pool["host"], pool["port"], pool["db"]) == ("example.test", 6380, 3)


def test_create_redis_fails_fast_instead_of_retrying(monkeypatch):
    """redis-py retries ten times by default; scoring cannot wait for that."""
    monkeypatch.setattr(settings, "redis_timeout_seconds", 0.5)
    client = create_redis("redis://127.0.0.1:6379/0")
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == kwargs["socket_connect_timeout"] == 0.5
    assert client.get_retry()._retries == 0


def test_a_malformed_url_is_rejected():
    with pytest.raises(ValueError, match="scheme"):
        create_redis("not-a-redis-url")


def test_ping_is_false_when_nothing_answers():
    client = create_redis(UNUSED_PORT_URL)
    started = time.monotonic()
    assert ping(client) is False
    # Refused instantly, not after a retry storm.
    assert time.monotonic() - started < 2
    client.close()


def test_ping_is_false_without_a_client():
    assert ping(None) is False


def test_open_configured_redis_returns_none_when_velocity_is_off(monkeypatch):
    monkeypatch.setattr(settings, "velocity_features", False)
    assert open_configured_redis() is None


def test_open_configured_redis_warns_but_starts_when_redis_is_down(monkeypatch, caplog):
    """A Redis that is down must not stop the service starting."""
    monkeypatch.setattr(settings, "velocity_features", True)
    monkeypatch.setattr(settings, "redis_url", UNUSED_PORT_URL)
    with caplog.at_level("WARNING"):
        client = open_configured_redis()
    assert client is not None
    assert "redis unavailable at startup" in caplog.text
    client.close()


# ------------------------------------------------------- against a real Redis
@pytest.mark.redis
def test_ping_reaches_a_running_redis(redis_client):
    assert ping(redis_client) is True


@pytest.mark.redis
def test_keys_round_trip_as_text(redis_client):
    """decode_responses: values come back as str, so nothing has to decode bytes."""
    redis_client.set("velocity:test", "card-1")
    assert redis_client.get("velocity:test") == "card-1"


@pytest.mark.redis
def test_a_connection_the_server_dropped_is_replaced(redis_client, redis_url):
    """Redis closing a connection must not cost a transaction its features.

    Redis drops idle connections, and restarts drop all of them. A command that
    has not been sent yet is safe to send again, so redis-py opens a new
    connection and the caller never sees the old one die. Turning retries off
    does not change that: it only stops a command retrying after it was sent.
    """
    redis_client.set("velocity:test", "1")
    connection_id = redis_client.client_id()

    killer = create_redis(redis_url)
    assert killer.client_kill_filter(_id=connection_id) == 1
    killer.close()

    assert redis_client.get("velocity:test") == "1"
    assert redis_client.client_id() != connection_id
