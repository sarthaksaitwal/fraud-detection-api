"""Isolation Forest: the unsupervised model from Steps 1.3 to 1.6.

Kept for the research notebooks (003 to 007). It is not the product model:
Step 1.7 showed a supervised XGBoost model is about 10x better on this data,
so the API serves src/ml/model.py instead.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.config import settings
from src.ml.preprocess import build_preprocessor
from src.ml.risk import RiskCalibrator, RiskModel


def build_model() -> Pipeline:
    """Unfitted preprocessor + Isolation Forest, bundled as one object.

    Bundling means the saved artifact carries its own preprocessing, so the API
    can never apply a different transformation than the one used in training.
    """
    forest = IsolationForest(
        n_estimators=settings.n_estimators,
        max_samples=settings.max_samples,
        contamination=settings.contamination,
        random_state=settings.random_seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", forest)])


def normal_only(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rows labelled normal (Class == 0): the model's picture of 'normal'."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    return X.loc[y == 0]


def fit_on_normal(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the whole pipeline on normal training transactions only.

    Fraud rows are held out of training entirely. The forest learns what normal
    looks like, and anything that does not fit that picture scores as anomalous.
    """
    X_normal = normal_only(X_train, y_train)
    if X_normal.empty:
        raise ValueError("no normal transactions to train on")
    return build_model().fit(X_normal)


def fit_risk_model(X_train: pd.DataFrame, y_train: pd.Series) -> RiskModel:
    """Fit the pipeline on normal rows, then calibrate risk against those rows.

    Reusing the training rows as the reference is sound when the training set is
    much larger than max_samples: each tree samples only 256 rows, so nearly
    every row is unseen by nearly every tree and its score is effectively
    out-of-sample. Notebook 004 checks this on held-out test data.
    """
    pipeline = fit_on_normal(X_train, y_train)
    reference = pipeline.score_samples(normal_only(X_train, y_train))
    return RiskModel(pipeline=pipeline, calibrator=RiskCalibrator.fit(reference))
