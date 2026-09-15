"""Step 1.3 guard rails: model construction and normal-only training."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.config import settings
from src.ml.isolation_forest import build_model, fit_on_normal, normal_only
from src.ml.preprocess import RAW_FEATURES, TARGET, split


@pytest.fixture
def fitted(raw_df):
    X_train, X_test, y_train, y_test = split(raw_df)
    return fit_on_normal(X_train, y_train), X_test, y_test


def test_build_model_bundles_preprocessing_and_forest():
    model = build_model()
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "model"]
    forest = model.named_steps["model"]
    assert isinstance(forest, IsolationForest)
    assert forest.n_estimators == settings.n_estimators
    assert forest.random_state == settings.random_seed


def test_normal_only_excludes_fraud(raw_df):
    X, y = raw_df[RAW_FEATURES], raw_df[TARGET]
    X_normal = normal_only(X, y)
    assert len(X_normal) == (y == 0).sum()
    assert y.loc[X_normal.index].eq(0).all()


def test_normal_only_rejects_misaligned_inputs(raw_df):
    X, y = raw_df[RAW_FEATURES], raw_df[TARGET]
    with pytest.raises(ValueError):
        normal_only(X, y.sample(frac=1, random_state=0))


def test_fit_on_normal_rejects_data_with_no_normal_rows(raw_df):
    fraud = raw_df[raw_df[TARGET] == 1]
    with pytest.raises(ValueError):
        fit_on_normal(fraud[RAW_FEATURES], fraud[TARGET])


def test_score_samples_returns_one_finite_score_per_row(fitted):
    model, X_test, _ = fitted
    scores = model.score_samples(X_test)
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()


def test_fraud_scores_lower_than_normal(fitted):
    model, X_test, y_test = fitted
    scores = model.score_samples(X_test)
    is_fraud = y_test.to_numpy() == 1
    assert scores[is_fraud].max() < np.median(scores[~is_fraud])


def test_training_is_reproducible(raw_df):
    X_train, X_test, y_train, _ = split(raw_df)
    first = fit_on_normal(X_train, y_train).score_samples(X_test)
    second = fit_on_normal(X_train, y_train).score_samples(X_test)
    np.testing.assert_array_equal(first, second)
