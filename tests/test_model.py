"""Guard rails for the XGBoost fraud model."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.ml.model import XGB_PARAMS, fit_xgboost, fraud_probability, out_of_fold_probability
from src.ml.preprocess import split


@pytest.fixture
def fitted(raw_df):
    X_train, X_test, y_train, y_test = split(raw_df)
    return fit_xgboost(X_train, y_train), X_train, X_test, y_train, y_test


def test_returns_a_pipeline_with_shared_preprocessing(fitted):
    model, *_ = fitted
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "model"]


def test_probabilities_are_valid(fitted):
    model, _, X_test, _, _ = fitted
    proba = fraud_probability(model, X_test)
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fraud_gets_higher_probability_than_normal(fitted):
    model, _, X_test, _, y_test = fitted
    proba = fraud_probability(model, X_test)
    is_fraud = y_test.to_numpy() == 1
    assert proba[is_fraud].min() > np.median(proba[~is_fraud])


def test_early_stopping_stops_before_the_tree_limit(fitted):
    model, *_ = fitted
    assert model.named_steps["model"].best_iteration < XGB_PARAMS["n_estimators"] - 1


def test_training_is_reproducible(raw_df):
    X_train, X_test, y_train, _ = split(raw_df)
    first = fraud_probability(fit_xgboost(X_train, y_train), X_test)
    second = fraud_probability(fit_xgboost(X_train, y_train), X_test)
    np.testing.assert_array_equal(first, second)


def test_out_of_fold_scores_every_training_row_once(raw_df):
    X_train, _, y_train, _ = split(raw_df)
    proba = out_of_fold_probability(X_train, y_train, n_splits=3)
    assert proba.shape == (len(X_train),)
    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()
    is_fraud = y_train.to_numpy() == 1
    assert np.median(proba[is_fraud]) > np.median(proba[~is_fraud])
