"""Step 1.4 guard rails: the 0-1 risk score."""

import pickle

import numpy as np
import pytest

from src.ml.preprocess import split
from src.ml.risk import RiskCalibrator
from src.ml.train import fit_risk_model

# Stand-in for the raw scores of normal traffic.
REFERENCE = np.linspace(-0.6, -0.4, 1001)


@pytest.fixture
def calibrator() -> RiskCalibrator:
    return RiskCalibrator.fit(REFERENCE[::-1])  # deliberately unsorted input


@pytest.fixture
def risk_model_and_test(raw_df):
    X_train, X_test, y_train, y_test = split(raw_df)
    return fit_risk_model(X_train, y_train), X_test, y_test


def test_reference_scores_are_stored_sorted(calibrator):
    assert np.all(np.diff(calibrator.reference_scores) >= 0)


def test_risk_is_bounded_even_far_outside_the_reference(calibrator):
    risk = calibrator.transform([-1e9, -0.5, 1e9])
    assert risk[0] == 1.0
    assert risk[2] == 0.0
    assert ((risk >= 0) & (risk <= 1)).all()


def test_lower_raw_score_never_means_lower_risk(calibrator):
    raw = np.linspace(-0.8, -0.2, 500)
    assert np.all(np.diff(calibrator.transform(raw)) <= 0)


def test_median_normal_score_is_mid_risk(calibrator):
    assert calibrator.transform([np.median(REFERENCE)])[0] == pytest.approx(0.5, abs=1e-3)


def test_threshold_equals_share_of_normal_traffic_flagged(calibrator):
    risk = calibrator.transform(REFERENCE)
    for threshold in (0.9, 0.99):
        assert (risk >= threshold).mean() == pytest.approx(1 - threshold, abs=2e-3)


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], []])
def test_fit_rejects_empty_or_non_finite_reference(bad):
    with pytest.raises(ValueError):
        RiskCalibrator.fit(bad)


def test_nan_raw_score_raises_instead_of_reading_as_safe(calibrator):
    with pytest.raises(ValueError):
        calibrator.transform([-0.5, np.nan])


def test_risk_model_is_bounded_and_points_the_right_way(risk_model_and_test):
    model, X_test, y_test = risk_model_and_test
    risk = model.risk(X_test)
    assert risk.shape == (len(X_test),)
    assert ((risk >= 0) & (risk <= 1)).all()
    is_fraud = y_test.to_numpy() == 1
    assert risk[is_fraud].min() > np.median(risk[~is_fraud])


def test_risk_preserves_the_raw_ranking(risk_model_and_test):
    model, X_test, _ = risk_model_and_test
    order = np.argsort(model.raw_scores(X_test))
    assert np.all(np.diff(model.risk(X_test)[order]) <= 0)


def test_risk_model_survives_pickling(risk_model_and_test):
    model, X_test, _ = risk_model_and_test
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.risk(X_test), restored.risk(X_test))
