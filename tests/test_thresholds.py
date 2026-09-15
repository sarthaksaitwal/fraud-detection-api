"""Step 1.6 guard rails: pricing policies and choosing thresholds."""

import numpy as np
import pandas as pd
import pytest

from src.ml.thresholds import (
    Costs,
    approve_all_cost_per_100k,
    choose_thresholds,
    policy_costs,
)

COSTS = Costs(review_cost=5.0, false_block_cost=50.0, chargeback_fee=15.0)

# Hand-checkable example: rows 4 and 5 are fraud worth $100 and $200.
Y = np.array([0, 0, 0, 0, 1, 1])
RISK = np.array([0.1, 0.5, 0.96, 0.995, 0.97, 0.999])
AMOUNTS = np.array([10.0, 10.0, 10.0, 10.0, 100.0, 200.0])
PER_100K = 100_000 / len(Y)


@pytest.fixture
def table() -> pd.DataFrame:
    return policy_costs(
        Y, RISK, AMOUNTS, COSTS, review_candidates=[0.95, 0.98], block_candidates=[0.99, None]
    )


def test_review_only_policy_costs(table):
    # review >= 0.95, never block: rows 2-5 reviewed at $5, no fraud missed.
    row = table[(table["review_threshold"] == 0.95) & table["block_threshold"].isna()].iloc[0]
    assert row["review_rate"] == pytest.approx(4 / 6)
    assert row["block_rate"] == 0
    assert row["missed_fraud_cost"] == 0
    assert row["cost_per_100k"] == pytest.approx(20 * PER_100K)


def test_review_and_block_policy_costs(table):
    # block >= 0.99: rows 3 (normal) and 5 (fraud). Nothing lands in [0.98, 0.99).
    # Row 4 (fraud, risk 0.97) is approved: $100 + $15 fee missed.
    row = table[(table["review_threshold"] == 0.98) & (table["block_threshold"] == 0.99)].iloc[0]
    assert row["review_cost"] == 0
    assert row["block_cost"] == pytest.approx(50 * PER_100K)
    assert row["missed_fraud_cost"] == pytest.approx(115 * PER_100K)
    assert row["fraud_stopped"] == pytest.approx(0.5)
    assert row["fraud_loss_stopped"] == pytest.approx(215 / 330)


def test_block_threshold_never_below_review_threshold():
    table = policy_costs(Y, RISK, AMOUNTS, COSTS)
    blocking = table.dropna(subset=["block_threshold"])
    assert (blocking["block_threshold"] >= blocking["review_threshold"]).all()


def test_approve_all_cost_is_every_fraud_amount_plus_fee():
    assert approve_all_cost_per_100k(Y, AMOUNTS, COSTS) == pytest.approx(330 * PER_100K)


def test_inputs_without_fraud_are_rejected():
    with pytest.raises(ValueError):
        policy_costs(np.zeros(6), RISK, AMOUNTS, COSTS)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        policy_costs(Y, RISK[:-1], AMOUNTS, COSTS)


CHOICES = pd.DataFrame(
    {
        "review_threshold": [0.98, 0.99, 0.99, 0.995],
        "block_threshold": [np.nan, np.nan, 0.999, np.nan],
        "review_rate": [0.021, 0.011, 0.010, 0.005],
        "block_rate": [0.0, 0.0, 0.001, 0.0],
        "cost_per_100k": [100.0, 100.5, 99.0, 130.0],
    }
)


def test_capacity_excludes_policies_that_review_too_much():
    choice = choose_thresholds(CHOICES, max_review_rate=0.02, tolerance=0.0)
    assert choice.review_threshold == 0.99
    assert choice.block_threshold == 0.999


def test_near_ties_go_to_the_policy_that_blocks_no_one():
    choice = choose_thresholds(CHOICES, max_review_rate=0.02, tolerance=0.02)
    assert choice.review_threshold == 0.99
    assert choice.block_threshold is None


def test_no_feasible_policy_raises():
    with pytest.raises(ValueError):
        choose_thresholds(CHOICES, max_review_rate=0.001)


@pytest.fixture
def overlapping():
    """Fraud risk higher than normal but overlapping: blocking would hit many customers."""
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.02).astype(int)
    risk = np.clip(rng.normal(0.95 + 0.03 * y, 0.02), 0, 1)
    amounts = rng.exponential(100, y.size)
    return y, risk, amounts


def test_expensive_false_blocks_mean_no_auto_block(overlapping):
    costs = Costs(review_cost=5.0, false_block_cost=1_000.0, chargeback_fee=15.0)
    choice = choose_thresholds(policy_costs(*overlapping, costs), max_review_rate=1.0)
    assert choice.block_threshold is None


def test_free_blocks_mean_blocking_replaces_review(overlapping):
    costs = Costs(review_cost=5.0, false_block_cost=0.0, chargeback_fee=15.0)
    choice = choose_thresholds(policy_costs(*overlapping, costs), max_review_rate=1.0)
    assert choice.block_threshold is not None
