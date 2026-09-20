"""Step 5.2 guard rails: the synthetic card, merchant and country."""

from collections import Counter
from inspect import signature

from src.api.schemas import Transaction
from src.features.entities import (
    CARD_COUNT,
    FOREIGN_COUNTRIES,
    HOME_COUNTRIES,
    MERCHANTS,
    card_for,
    country_for,
    entities_for,
    merchant_for,
)
from src.ml.preprocess import RAW_FEATURES

SAMPLE_IDS = [f"test-row-{row}" for row in range(20_000)]


def test_the_same_transaction_always_gets_the_same_identity():
    """The producer resumes with --start, and Kafka redelivers: ids must be stable."""
    assert entities_for("test-row-7") == entities_for("test-row-7")
    assert entities_for("test-row-7") != entities_for("test-row-8")


def test_identities_are_stable_across_processes():
    """Not Python's hash(), which is salted per process and would change every run."""
    assert entities_for("test-row-7") == {
        "card_id": "card-00370",
        "merchant": "online-marketplace",
        "country": "IN",
    }


def test_every_generated_value_is_accepted_by_the_schema(raw_df):
    values = raw_df[RAW_FEATURES].iloc[0].to_dict()
    for transaction_id in SAMPLE_IDS[:500]:
        transaction = Transaction(
            transaction_id=transaction_id, **entities_for(transaction_id), **values
        )
        assert transaction.card_id and transaction.merchant and transaction.country


def test_card_traffic_is_heavy_tailed():
    """Spread evenly, no card would ever be busy enough for velocity to notice."""
    counts = Counter(card_for(transaction_id) for transaction_id in SAMPLE_IDS)
    busiest = counts.most_common(len(counts) // 100)
    share = sum(count for _, count in busiest) / len(SAMPLE_IDS)
    assert 0.05 < share < 0.20  # the top 1% of cards take about a tenth of the traffic
    assert counts.most_common(1)[0][1] < len(SAMPLE_IDS) * 0.02  # but no card dominates


def test_cards_come_from_the_whole_portfolio():
    cards = {card_for(transaction_id) for transaction_id in SAMPLE_IDS}
    assert len(cards) > CARD_COUNT * 0.5
    assert all(card.startswith("card-") for card in cards)


def test_merchants_are_spread_over_the_list():
    merchants = Counter(merchant_for(transaction_id) for transaction_id in SAMPLE_IDS)
    assert set(merchants) == set(MERCHANTS)


def test_a_card_has_one_home_country_it_usually_transacts_in():
    at_home = [country_for(transaction_id, "card-00001") for transaction_id in SAMPLE_IDS[:1000]]
    home = Counter(at_home).most_common(1)[0][0]
    assert home in HOME_COUNTRIES
    assert at_home.count(home) > 900  # a few transactions happen elsewhere


def test_some_transactions_happen_abroad():
    """A card suddenly transacting in two countries is what velocity looks for."""
    countries = {
        country_for(transaction_id, card_for(transaction_id)) for transaction_id in SAMPLE_IDS
    }
    assert countries & set(FOREIGN_COUNTRIES)


def test_identities_never_see_the_fraud_label():
    """The generator takes a transaction id and nothing else, so it cannot be rigged.

    Put fraud on a handful of cards and velocity would "catch" it, measuring the
    generator rather than the feature.
    """
    assert list(signature(entities_for).parameters) == ["transaction_id"]
