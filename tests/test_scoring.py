"""Step 2.2 guard rails: scores and decisions."""

import logging

import numpy as np
import pytest

from src.api.schemas import Decision, Transaction
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
