"""Step 5.3 guard rails: counting recent activity per card in Redis."""

import pytest

from src.api.schemas import Transaction
from src.features.velocity import (
    VelocityStore,
    features_from,
    key_for,
    member_for,
)
from src.ml.preprocess import RAW_FEATURES

NOW = 1_800_000_000.0


@pytest.fixture
def values(raw_df):
    """The model columns of one row, to build transactions from."""
    return raw_df[RAW_FEATURES].iloc[0].to_dict()


@pytest.fixture
def make_transaction(values):
    def make(transaction_id, card_id="card-00001", amount=10.0, country="GB"):
        return Transaction(
            **{**values, "Amount": amount},
            transaction_id=transaction_id,
            card_id=card_id,
            country=country,
        )

    return make


# ------------------------------------------------------- turning events into features
def event(transaction_id, seconds_ago, amount=10.0, country="GB"):
    return (f"{transaction_id}|{amount:.2f}|{country}", NOW - seconds_ago)


def test_each_window_counts_only_what_falls_inside_it():
    events = [event("a", 10), event("b", 120), event("c", 1800), event("d", 3599)]
    features = features_from(events, NOW, current="a|10.00|GB")
    assert (features.count_1m, features.count_5m, features.count_1h) == (1, 2, 4)


def test_amounts_and_countries_come_from_the_same_read():
    events = [event("a", 5, amount=10), event("b", 30, amount=2.50, country="NG")]
    features = features_from(events, NOW, current="a|10.00|GB")
    assert features.amount_1h == 12.50
    assert features.countries_1h == 2


def test_time_since_the_previous_transaction_ignores_this_one():
    events = [event("a", 0), event("b", 42)]
    features = features_from(events, NOW, current="a|10.00|GB")
    assert features.seconds_since_previous == 42


def test_a_cards_first_transaction_has_no_previous_one():
    features = features_from([event("a", 0)], NOW, current="a|10.00|GB")
    assert features.seconds_since_previous is None
    assert features.count_1m == 1


def test_a_transaction_id_containing_the_separator_still_parses():
    events = [("od|d|id|10.00|GB", NOW)]
    assert features_from(events, NOW, current="").amount_1h == 10.0


def test_the_member_carries_what_the_features_need(make_transaction):
    assert member_for(make_transaction("tx-1", amount=12.5)) == "tx-1|12.50|GB"
    assert key_for("card-00001") == "vel:card:card-00001"


# ----------------------------------------------------------- against a real Redis
@pytest.mark.redis
def test_a_first_transaction_counts_itself_and_nothing_else(redis_client, make_transaction):
    store = VelocityStore(redis_client)
    (features,) = store.measure([make_transaction("tx-1")])
    assert (features.count_1m, features.count_1h) == (1, 1)
    assert features.seconds_since_previous is None
    assert features.amount_1h == 10.0


@pytest.mark.redis
def test_a_card_used_again_sees_its_own_history(redis_client, make_transaction):
    store = VelocityStore(redis_client)
    store.measure([make_transaction("tx-1", amount=10)], now=NOW)
    store.measure([make_transaction("tx-2", amount=20)], now=NOW + 30)
    (features,) = store.measure([make_transaction("tx-3", amount=5)], now=NOW + 45)
    assert features.count_1m == 3
    assert features.amount_1h == 35.0
    assert features.seconds_since_previous == 15


@pytest.mark.redis
def test_cards_are_counted_separately(redis_client, make_transaction):
    store = VelocityStore(redis_client)
    store.measure([make_transaction("tx-1", card_id="card-00001")])
    (features,) = store.measure([make_transaction("tx-2", card_id="card-00002")])
    assert features.count_1h == 1


@pytest.mark.redis
def test_a_transaction_without_a_card_has_no_features(redis_client, values):
    store = VelocityStore(redis_client)
    anonymous = Transaction(**values, transaction_id="tx-1")
    assert store.measure([anonymous]) == [None]
    assert redis_client.dbsize() == 0


@pytest.mark.redis
def test_transactions_in_one_batch_see_each_other(redis_client, make_transaction):
    """The server runs a pipeline in order, so the second counts the first."""
    store = VelocityStore(redis_client)
    batch = [make_transaction(f"tx-{i}") for i in range(3)]
    assert [f.count_1m for f in store.measure(batch)] == [1, 2, 3]


@pytest.mark.redis
def test_a_batch_costs_one_round_trip(redis_client, make_transaction):
    """500 transactions must not mean 500 conversations with Redis."""
    store = VelocityStore(redis_client)
    pipelines = []
    original = redis_client.pipeline
    redis_client.pipeline = lambda *a, **k: pipelines.append(original(*a, **k)) or pipelines[-1]

    store.measure([make_transaction(f"tx-{i}", card_id=f"card-{i:05d}") for i in range(500)])
    assert len(pipelines) == 1


@pytest.mark.redis
def test_a_redelivered_transaction_is_not_counted_twice(redis_client, make_transaction):
    """Kafka delivers at least once; the same member updates instead of duplicating."""
    store = VelocityStore(redis_client)
    store.measure([make_transaction("tx-1")], now=NOW)
    (features,) = store.measure([make_transaction("tx-1")], now=NOW + 5)
    assert features.count_1m == 1
    assert features.seconds_since_previous is None


@pytest.mark.redis
def test_events_older_than_the_window_are_dropped(redis_client, make_transaction):
    store = VelocityStore(redis_client, retention=60)
    store.measure([make_transaction("tx-1")], now=NOW)
    (features,) = store.measure([make_transaction("tx-2")], now=NOW + 61)
    assert features.count_1h == 1
    assert store.card_count("card-00001") == 1  # the old one is gone from Redis, not just ignored


@pytest.mark.redis
def test_no_card_can_grow_without_limit(redis_client, make_transaction):
    """A card stuck in a loop must not fill Redis; the newest events win."""
    store = VelocityStore(redis_client, max_events=5)
    features = store.measure([make_transaction(f"tx-{i}") for i in range(20)])
    assert store.card_count("card-00001") == 5
    assert features[-1].count_1m == 5


@pytest.mark.redis
def test_a_card_that_goes_quiet_expires(redis_client, make_transaction):
    store = VelocityStore(redis_client, retention=120)
    store.measure([make_transaction("tx-1")])
    assert 0 < redis_client.ttl(key_for("card-00001")) <= 120
