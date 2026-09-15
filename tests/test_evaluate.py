"""Step 1.5 guard rails: metrics for an imbalanced risk score."""

import numpy as np
import pytest

from src.ml.evaluate import bootstrap_pr_auc, confusion_at, summary_metrics, threshold_table

# Hand-checkable example. At threshold 0.9 rows 1, 2 and 5 are flagged:
#   rows 2, 5 are fraud -> tp = 2    row 1 is normal -> fp = 1
#   row 3 is missed fraud -> fn = 1  rows 0, 4 are quiet normals -> tn = 2
Y = np.array([0, 0, 1, 1, 0, 1])
RISK = np.array([0.1, 0.95, 0.99, 0.5, 0.2, 0.999])
AMOUNTS = np.array([10.0, 20.0, 300.0, 100.0, 5.0, 600.0])


@pytest.fixture
def noisy():
    """20,000 rows, 5% fraud, fraud risk shifted up but overlapping normal risk."""
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.05).astype(int)
    risk = np.clip(rng.normal(0.5 + 0.2 * y, 0.15), 0, 1)
    return y, risk


def test_confusion_counts_match_hand_example():
    assert confusion_at(Y, RISK, 0.9) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_threshold_table_precision_recall_and_money():
    row = threshold_table(Y, RISK, thresholds=[0.9], amounts=AMOUNTS).iloc[0]
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["alerts_per_fraud"] == pytest.approx(1.5)
    assert row["normal_flagged_%"] == pytest.approx(100 / 3)
    assert row["fraud_amount_caught_%"] == pytest.approx(100 * 900 / 1000)


def test_precision_is_undefined_when_nothing_is_flagged():
    row = threshold_table(Y, RISK, thresholds=[1.5]).iloc[0]
    assert row["flagged"] == 0
    assert np.isnan(row["precision"])
    assert row["recall"] == 0


def test_recall_never_rises_as_threshold_rises(noisy):
    table = threshold_table(*noisy, thresholds=np.linspace(0, 1, 21))
    assert np.all(np.diff(table["recall"].to_numpy()) <= 0)


def test_perfect_ranking_scores_one():
    metrics = summary_metrics(Y, Y.astype(float))
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_random_ranking_pr_auc_is_close_to_fraud_rate():
    rng = np.random.default_rng(1)
    y = (rng.random(50_000) < 0.05).astype(int)
    metrics = summary_metrics(y, rng.random(y.size))
    assert metrics["pr_auc"] == pytest.approx(metrics["pr_auc_baseline"], abs=0.01)
    assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    ("y", "risk"),
    [
        ([0, 1, 1], [0.1, 0.2]),  # length mismatch
        ([0, 0, 0], [0.1, 0.2, 0.3]),  # no fraud at all
        ([0, 2, 1], [0.1, 0.2, 0.3]),  # not a 0/1 label
    ],
)
def test_invalid_inputs_are_rejected(y, risk):
    with pytest.raises(ValueError):
        summary_metrics(y, risk)


def test_bootstrap_interval_brackets_the_point_estimate(noisy):
    low, high = bootstrap_pr_auc(*noisy, n_resamples=100, seed=0)
    point = summary_metrics(*noisy)["pr_auc"]
    assert 0 <= low <= point <= high <= 1


def test_bootstrap_is_reproducible_with_a_seed(noisy):
    assert bootstrap_pr_auc(*noisy, n_resamples=30, seed=7) == bootstrap_pr_auc(
        *noisy, n_resamples=30, seed=7
    )
