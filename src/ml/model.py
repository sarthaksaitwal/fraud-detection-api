"""XGBoost fraud model: the model the API serves.

Introduced as a baseline in Step 1.7, where it beat the unsupervised Isolation
Forest by about 10x PR-AUC, and then adopted as the product model. It trains on
fraud labels and outputs a fraud probability, which is used directly as risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import settings
from src.ml.preprocess import build_preprocessor

# Moderate, commonly used settings. Early stopping decides the number of trees.
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 100,
    "tree_method": "hist",
}
VALIDATION_SIZE = 0.2


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit on training data *with* fraud labels.

    A stratified slice of the training data is held back to decide when to stop
    adding trees, so the test set stays untouched until evaluation.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=settings.random_seed,
    )
    preprocessor = build_preprocessor().fit(X_fit)
    model = XGBClassifier(**XGB_PARAMS, random_state=settings.random_seed, n_jobs=-1)
    model.fit(
        preprocessor.transform(X_fit),
        y_fit,
        eval_set=[(preprocessor.transform(X_val), y_val)],
        verbose=False,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fraud_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability that each transaction is fraud."""
    return model.predict_proba(X)[:, 1]


def out_of_fold_probability(
    X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5
) -> np.ndarray:
    """Fraud probability for every training row, from a model that never saw it.

    A model re-scoring its own training rows is far too confident, so those
    scores cannot be used to choose thresholds. Instead the training data is cut
    into `n_splits` folds and each fold is scored by a model fit on the others.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.random_seed)
    proba = np.full(len(X_train), np.nan)
    for fit_rows, score_rows in folds.split(X_train, y_train):
        fold_model = fit_xgboost(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
        proba[score_rows] = fraud_probability(fold_model, X_train.iloc[score_rows])
    return proba
