"""Feature preparation shared by training and serving.

Everything that turns a raw transaction into model input lives here, so the
training script and the API apply *exactly* the same transformation. If the two
ever drifted apart, the model would silently score garbage (training/serving skew).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.config import settings

TARGET = "Class"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_COLUMNS, "Amount"]
SECONDS_PER_DAY = 24 * 60 * 60


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the Kaggle CSV and fail loudly if its schema is not what we expect."""
    df = pd.read_csv(path or settings.raw_data_path)
    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows so no transaction can land in both train and test."""
    return df.drop_duplicates().reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split: both halves keep the same fraud ratio."""
    return train_test_split(
        df[RAW_FEATURES],
        df[TARGET],
        test_size=settings.test_size,
        stratify=df[TARGET],
        random_state=settings.random_seed,
    )


def time_to_cyclical(time: pd.DataFrame) -> np.ndarray:
    """Map seconds to a position on a 24h clock, encoded as (sin, cos).

    As a plain number, 23:59 and 00:01 look maximally far apart. On the circle
    they are neighbours, which is the truth.
    """
    seconds_into_day = np.asarray(time, dtype=float).ravel() % SECONDS_PER_DAY
    angle = 2 * np.pi * seconds_into_day / SECONDS_PER_DAY
    return np.column_stack([np.sin(angle), np.cos(angle)])


def cyclical_feature_names(transformer, input_features) -> list[str]:
    return ["time_sin", "time_cos"]


def build_preprocessor() -> ColumnTransformer:
    """Unfitted transformer: raw columns in, model features out.

    Fit it on training data only, then reuse that one fitted object everywhere.
    """
    amount = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )
    time_of_day = FunctionTransformer(time_to_cyclical, feature_names_out=cyclical_feature_names)

    return ColumnTransformer(
        [
            ("amount", amount, ["Amount"]),
            ("time", time_of_day, ["Time"]),
            ("v", "passthrough", V_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def save_splits(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Persist the raw (unscaled) splits so every later step uses identical rows."""
    settings.ensure_dirs()
    X_train.assign(**{TARGET: y_train}).to_parquet(settings.train_split_path)
    X_test.assign(**{TARGET: y_test}).to_parquet(settings.test_split_path)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_parquet(settings.train_split_path)
    test = pd.read_parquet(settings.test_split_path)
    return train[RAW_FEATURES], test[RAW_FEATURES], train[TARGET], test[TARGET]
