"""Step 5.5 guard rails: when velocity overrides the model's decision."""

import pytest

from src.api.schemas import Decision, VelocityFeatures
from src.config import settings
from src.features.rules import describe, escalate, reasons_for, velocity_rules


def velocity(**overrides):
    """Features for a quiet card, unless a test says otherwise."""
    quiet = {
        "count_1m": 1,
        "count_5m": 1,
        "count_1h": 1,
        "amount_1h": 10.0,
        "countries_1h": 1,
        "seconds_since_previous": None,
    }
    return VelocityFeatures(**(quiet | overrides))


def test_a_quiet_card_fires_nothing():
    assert reasons_for(velocity()) == []


@pytest.mark.parametrize(
    "features, rule",
    [
        ({"count_1m": 6}, "many_in_a_minute"),
        ({"count_1h": 11}, "many_in_an_hour"),
        ({"countries_1h": 3}, "several_countries"),
        ({"seconds_since_previous": 1.9}, "back_to_back"),
    ],
)
def test_each_rule_fires_just_past_its_threshold(features, rule):
    assert reasons_for(velocity(**features)) == [rule]


@pytest.mark.parametrize(
    "features",
    [
        {"count_1m": 5},
        {"count_1h": 10},
        {"countries_1h": 2},
        {"seconds_since_previous": 2.0},
    ],
)
def test_nothing_fires_at_the_threshold_itself(features):
    """The thresholds are "more than", so a card exactly at one is still fine."""
    assert reasons_for(velocity(**features)) == []


def test_several_rules_can_fire_at_once():
    busy = velocity(count_1m=9, count_1h=40, countries_1h=4, seconds_since_previous=0.2)
    assert reasons_for(busy) == [
        "many_in_a_minute",
        "many_in_an_hour",
        "several_countries",
        "back_to_back",
    ]


def test_no_velocity_means_no_reasons():
    """Redis down, or a transaction with no card: the model decides alone."""
    assert reasons_for(None) == []


def test_thresholds_come_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "velocity_max_per_minute", 2)
    assert reasons_for(velocity(count_1m=3)) == ["many_in_a_minute"]


def test_an_approved_transaction_is_escalated_to_review():
    assert escalate(Decision.APPROVE, ["many_in_a_minute"]) is Decision.REVIEW


def test_rules_never_block():
    """A wrong block costs a customer their payment; velocity is a suspicion, not proof."""
    every_rule = [rule.name for rule in velocity_rules()]
    assert escalate(Decision.APPROVE, every_rule) is Decision.REVIEW


@pytest.mark.parametrize("decision", [Decision.REVIEW, Decision.BLOCK])
def test_a_decision_the_model_already_made_is_left_alone(decision):
    assert escalate(decision, ["many_in_a_minute"]) is decision


def test_escalation_needs_a_reason():
    assert escalate(Decision.APPROVE, []) is Decision.APPROVE


def test_every_rule_explains_itself():
    """An analyst opening a review sees why it was sent, not a rule name."""
    for rule in velocity_rules():
        assert describe(rule.name) == rule.description
    assert "more than 5 transactions" in describe("many_in_a_minute")
    assert describe("unknown_rule") == "unknown_rule"
