"""What a card has done recently, counted in Redis (Step 5.3).

One key per card, a sorted set, scored by arrival time:

    vel:card:card-00042  ->  { "test-row-5|12.50|GB": 1758349021.4, ... }

A sorted set is the right shape because every velocity question is "what falls
in this time range?", which is a range query over the score. Members carry the
amount and country as well as the transaction id, so one read answers all of
them instead of one read per feature.

Each measurement is five commands sent as one pipeline, so a card's features
cost a single round trip:

    ZREMRANGEBYSCORE  drop what has aged out of the window
    ZADD              add this transaction
    ZREMRANGEBYRANK   keep at most MAX_EVENTS_PER_CARD, so no card can grow without limit
    ZRANGEBYSCORE     read the window back, with scores
    EXPIRE            a card that goes quiet disappears on its own

A whole batch goes in one pipeline too, and the server runs the commands in
order, so two transactions on the same card in one batch still see each other.

**Time is arrival time, not the dataset's `Time` column.** The producer replays
two days of transactions in minutes, so the dataset's own clock would put every
transaction in the same window. Velocity measures the traffic the service is
actually seeing, which is what a real system does.

A redelivered transaction re-adds the identical member, and a sorted set stores
each member once, so its score is updated rather than counted twice.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence

from redis import Redis
from redis.client import Pipeline
from redis.exceptions import RedisError

from src.api.schemas import Transaction, VelocityFeatures

log = logging.getLogger("src.features.velocity")

KEY_PREFIX = "vel:card:"
# Members are "<transaction_id>|<amount>|<country>". A transaction id may itself
# contain "|", so members are split from the right.
FIELD_SEPARATOR = "|"

# The windows reported per card, in seconds, longest last. The longest is also
# how long events are kept: nothing older can affect a feature.
WINDOWS_SECONDS = (60, 300, 3600)
RETENTION_SECONDS = WINDOWS_SECONDS[-1]

# The most events kept per card. A card at this many in an hour is already far
# past any velocity threshold, and the cap bounds both Redis memory and the work
# of reading a window back.
MAX_EVENTS_PER_CARD = 256


def member_for(transaction: Transaction) -> str:
    """The sorted-set member for a transaction: everything a feature needs from it."""
    return FIELD_SEPARATOR.join(
        [transaction.transaction_id, f"{transaction.Amount:.2f}", transaction.country or ""]
    )


def key_for(card_id: str) -> str:
    return f"{KEY_PREFIX}{card_id}"


def features_from(
    events: Sequence[tuple[str, float]], now: float, current: str
) -> VelocityFeatures:
    """Turn one card's recent events into features.

    events: (member, arrival time) pairs, as ZRANGEBYSCORE returns them.
    current: the member for the transaction being scored, so that "time since
    the previous transaction" does not measure the gap to itself.
    """
    counts = dict.fromkeys(WINDOWS_SECONDS, 0)
    amount, countries, previous = 0.0, set(), None
    for member, scored_at in events:
        for window in WINDOWS_SECONDS:
            if scored_at >= now - window:
                counts[window] += 1
        _, spent, country = member.rsplit(FIELD_SEPARATOR, 2)
        amount += float(spent)
        if country:
            countries.add(country)
        if member != current and (previous is None or scored_at > previous):
            previous = scored_at
    return VelocityFeatures(
        count_1m=counts[60],
        count_5m=counts[300],
        count_1h=counts[3600],
        amount_1h=round(amount, 2),
        countries_1h=len(countries),
        seconds_since_previous=None if previous is None else round(now - previous, 3),
    )


class VelocityStore:
    """Records transactions in Redis and reads back what each card has been doing."""

    def __init__(
        self,
        client: Redis,
        retention: float = RETENTION_SECONDS,
        max_events: int = MAX_EVENTS_PER_CARD,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self._retention = retention
        self._max_events = max_events
        self._clock = clock

    def measure(
        self, transactions: Iterable[Transaction], now: float | None = None
    ) -> list[VelocityFeatures | None]:
        """Record each transaction and return its card's features, in the same order.

        One round trip for the whole batch. A transaction with no card_id gets
        None: there is nothing to count it against.

        Raises RedisError if Redis is unreachable; the caller decides whether
        that costs the transaction or only its features (Step 5.6).
        """
        transactions = list(transactions)
        now = self._clock() if now is None else now
        identified = [t for t in transactions if t.card_id]
        if not identified:
            return [None] * len(transactions)

        pipeline = self.client.pipeline(transaction=False)
        for transaction in identified:
            self._queue(pipeline, transaction, now)
        replies = pipeline.execute()

        # Five commands per transaction; the window is the fourth reply of each.
        windows = iter(replies[3::5])
        features = {
            t.transaction_id: features_from(next(windows), now, member_for(t)) for t in identified
        }
        return [features.get(t.transaction_id) if t.card_id else None for t in transactions]

    def _queue(self, pipeline: Pipeline, transaction: Transaction, now: float) -> None:
        key = key_for(transaction.card_id)
        cutoff = now - self._retention
        pipeline.zremrangebyscore(key, "-inf", cutoff)
        pipeline.zadd(key, {member_for(transaction): now})
        # Ranks are oldest first, so this drops everything but the newest max_events.
        pipeline.zremrangebyrank(key, 0, -self._max_events - 1)
        pipeline.zrangebyscore(key, cutoff, "+inf", withscores=True)
        pipeline.expire(key, math.ceil(self._retention))

    def measure_or_none(self, transactions: Iterable[Transaction]) -> list[VelocityFeatures | None]:
        """`measure`, but a Redis failure costs the features rather than the transaction.

        Velocity is a signal, not a gate: refusing a payment because a cache is
        down would be a worse outage than scoring without the signal. Step 5.6
        adds a resting window, so a Redis that is down is not retried on every
        transaction.
        """
        transactions = list(transactions)
        try:
            return self.measure(transactions)
        except RedisError as exc:
            log.warning(
                "velocity unavailable, scoring %d transaction(s) without it: %s: %s",
                len(transactions),
                type(exc).__name__,
                exc,
            )
            return [None] * len(transactions)

    def card_count(self, card_id: str) -> int:
        """Events currently kept for a card. For tests and the dashboard, not scoring."""
        return int(self.client.zcard(key_for(card_id)))


def open_configured_velocity(client: Redis | None) -> VelocityStore | None:
    """A store on this Redis client, or None when velocity features are switched off."""
    return None if client is None else VelocityStore(client)
