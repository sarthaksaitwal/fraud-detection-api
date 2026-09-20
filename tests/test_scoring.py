"""Step 2.2 guard rails: scores and decisions."""

import logging

import numpy as np
import pytest

from src.api.schemas import Decision, Transaction, VelocityFeatures
from src.ml.artifact import save_model
from src.ml.model import fit_xgboost, fraud_probability
from src.ml.preprocess import RAW_FEATURES, split
from src.scoring import Scorer, decide


class FixedProbability:
    """Stands in for the model: returns preset fraud probabilities, in order."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_proba(self, X):
        assert list(X.columns) == RAW_FEATURES
        p = self.probabilities[: len(X)]
        return np.column_stack([1 - p, p])


@pytest.fixture
def transactions(raw_df):
    rows = raw_df[RAW_FEATURES].head(3).to_dict(orient="records")
    return [Transaction(**row, transaction_id=f"tx-{i}") for i, row in enumerate(rows)]


@pytest.fixture
def saved_model(raw_df, tmp_path):
    X_train, X_test, y_train, _ = split(raw_df)
    pipeline = fit_xgboost(X_train, y_train)
    metadata = {"decision_thresholds": {"review": 0.24, "block": 0.95}}
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    saved = save_model(pipeline, metadata, model_path, metadata_path)
    return model_path, metadata_path, pipeline, saved, X_test


def test_decisions_at_and_around_the_thresholds():
    risk = np.array([0.0, 0.2399, 0.24, 0.9499, 0.95, 1.0])
    decisions = decide(risk, 0.24, 0.95)
    assert all(type(d) is Decision for d in decisions)
    assert decisions == [
        Decision.APPROVE,
        Decision.APPROVE,
        Decision.REVIEW,
        Decision.REVIEW,
        Decision.BLOCK,
        Decision.BLOCK,
    ]


def test_no_block_threshold_means_never_block():
    assert decide(np.array([0.5, 1.0]), 0.24, None) == [Decision.REVIEW, Decision.REVIEW]


def test_results_keep_order_ids_and_explain_the_decision(transactions):
    scorer = Scorer(FixedProbability([0.01, 0.5, 0.99]), "v-test", 0.24, 0.95)
    results = scorer.score(transactions)
    assert [r.transaction_id for r in results] == ["tx-0", "tx-1", "tx-2"]
    assert [r.decision for r in results] == [Decision.APPROVE, Decision.REVIEW, Decision.BLOCK]
    assert results[1].risk_score == pytest.approx(0.5)
    assert {(r.review_threshold, r.block_threshold, r.model_version) for r in results} == {
        (0.24, 0.95, "v-test")
    }


def test_empty_input_needs_no_model_call():
    scorer = Scorer(FixedProbability([]), "v-test", 0.24, 0.95)
    assert scorer.score([]) == []


@pytest.mark.parametrize("review, block", [(-0.1, None), (1.1, None), (0.5, 0.4), (0.2, 1.5)])
def test_invalid_thresholds_are_refused(review, block):
    with pytest.raises(ValueError):
        Scorer(FixedProbability([]), "v-test", review, block)


def test_loaded_scorer_matches_the_model(saved_model):
    model_path, metadata_path, pipeline, saved, X_test = saved_model
    scorer = Scorer.load(model_path, metadata_path, review_threshold=0.24, block_threshold=0.95)
    batch = [Transaction(**row) for row in X_test.to_dict(orient="records")]
    results = scorer.score(batch)
    np.testing.assert_allclose(
        [r.risk_score for r in results], fraud_probability(pipeline, X_test), rtol=1e-6
    )
    assert results[0].model_version == saved["model_version"]


def test_batch_and_single_scoring_agree(saved_model):
    model_path, metadata_path, *_, X_test = saved_model
    scorer = Scorer.load(model_path, metadata_path)
    batch = [Transaction(**row) for row in X_test.head(5).to_dict(orient="records")]
    together = scorer.score(batch)
    one_by_one = [scorer.score_one(t) for t in batch]
    assert [r.risk_score for r in together] == pytest.approx([r.risk_score for r in one_by_one])
    assert [r.decision for r in together] == [r.decision for r in one_by_one]


def test_load_warns_when_thresholds_differ_from_training(saved_model, caplog):
    model_path, metadata_path, *_ = saved_model
    with caplog.at_level(logging.WARNING, logger="src.scoring"):
        Scorer.load(model_path, metadata_path, review_threshold=0.5, block_threshold=0.95)
    assert "trained with review=0.24" in caplog.text


def test_load_is_quiet_when_thresholds_match(saved_model, caplog):
    model_path, metadata_path, *_ = saved_model
    with caplog.at_level(logging.WARNING, logger="src.scoring"):
        Scorer.load(model_path, metadata_path, review_threshold=0.24, block_threshold=0.95)
    assert caplog.text == ""


# --------------------------------------------------------- velocity (Step 5.4)
VELOCITY = VelocityFeatures(
    count_1m=4, count_5m=6, count_1h=9, amount_1h=90.0, countries_1h=2, seconds_since_previous=1.5
)


def test_velocity_is_carried_into_the_result(transactions):
    """The model is not given it -- it was trained without it -- but the result explains itself."""
    scorer = Scorer(FixedProbability([0.1, 0.5, 0.9]), "v-test", 0.24, 0.95)
    identified = [t.model_copy(update={"card_id": "card-00001"}) for t in transactions]
    features = [VELOCITY] + [None] * (len(identified) - 1)
    results = scorer.score(identified, features)
    assert results[0].velocity == VELOCITY
    assert results[0].card_id == "card-00001"
    assert results[1].velocity is None


def test_scoring_without_velocity_is_unchanged(transactions):
    scorer = Scorer(FixedProbability([0.1, 0.5, 0.9]), "v-test", 0.24, 0.95)
    without = scorer.score(transactions)
    with_none = scorer.score(transactions, [None] * len(transactions))
    assert [r.risk_score for r in without] == [r.risk_score for r in with_none]
    assert all(r.velocity is None for r in without)


def test_velocity_must_line_up_with_the_transactions(transactions):
    """A mismatch would attach one card's history to another card's transaction."""
    scorer = Scorer(FixedProbability([0.1, 0.5, 0.9]), "v-test", 0.24, 0.95)
    with pytest.raises(ValueError):
        scorer.score(transactions, [VELOCITY])


# ------------------------------------------------------- the rules (Step 5.5)
BUSY = VelocityFeatures(
    count_1m=9, count_5m=9, count_1h=9, amount_1h=90.0, countries_1h=1, seconds_since_previous=30.0
)


def test_a_busy_card_turns_an_approval_into_a_review(transactions):
    """The model sees one transaction; the rules see what the card has been doing."""
    scorer = Scorer(FixedProbability([0.01, 0.01, 0.01]), "v-test", 0.24, 0.95)
    results = scorer.score(transactions, [BUSY, None, None])
    assert results[0].decision is Decision.REVIEW
    assert results[0].model_decision is Decision.APPROVE
    assert results[0].reasons == ["many_in_a_minute"]
    # The risk score is the model's and is not touched by the rules.
    assert results[0].risk_score == pytest.approx(0.01)
    assert results[1].decision is Decision.APPROVE
    assert results[1].reasons == []


def test_the_rules_never_soften_a_block(transactions):
    scorer = Scorer(FixedProbability([0.99, 0.5, 0.01]), "v-test", 0.24, 0.95)
    blocked, reviewed, _ = scorer.score(transactions, [BUSY, BUSY, None])
    assert (blocked.decision, blocked.model_decision) == (Decision.BLOCK, Decision.BLOCK)
    assert (reviewed.decision, reviewed.model_decision) == (Decision.REVIEW, Decision.REVIEW)
    # The rules fired; they simply had nothing to add.
    assert blocked.reasons == ["many_in_a_minute"]


def test_without_velocity_the_model_decides_alone(transactions):
    scorer = Scorer(FixedProbability([0.01, 0.01, 0.01]), "v-test", 0.24, 0.95)
    result, *_ = scorer.score(transactions)
    assert result.decision is result.model_decision is Decision.APPROVE
    assert result.reasons == []
