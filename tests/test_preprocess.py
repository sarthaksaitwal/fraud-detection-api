"""Step 1.2 guard rails: dedup, stratified split, and the preprocessor contract."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.ml.preprocess import (
    RAW_FEATURES,
    TARGET,
    V_COLUMNS,
    build_preprocessor,
    deduplicate,
    split,
    time_to_cyclical,
)


def test_deduplicate_removes_exact_copies(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(10)])
    assert len(deduplicate(doubled)) == len(raw_df)


def test_split_is_stratified(raw_df):
    X_train, X_test, y_train, y_test = split(raw_df)
    assert list(X_train.columns) == RAW_FEATURES
    assert y_train.mean() == pytest.approx(raw_df[TARGET].mean())
    assert y_test.mean() == pytest.approx(raw_df[TARGET].mean())


def test_split_is_reproducible(raw_df):
    assert split(raw_df)[1].index.equals(split(raw_df)[1].index)


def test_preprocessor_output_shape_and_names(raw_df):
    X_train, X_test, *_ = split(raw_df)
    out = build_preprocessor().fit(X_train).transform(X_test)
    assert out.shape == (len(X_test), 31)
    assert set(out.columns) == {"Amount", "time_sin", "time_cos", *V_COLUMNS}
    assert not out.isna().any().any()


def test_preprocessor_drops_unknown_columns(raw_df):
    pre = build_preprocessor().fit(raw_df[RAW_FEATURES])
    out = pre.transform(raw_df[RAW_FEATURES].assign(card_id="abc"))
    assert "card_id" not in out.columns


def test_cyclical_time_wraps_midnight():
    just_before, just_after = time_to_cyclical(pd.DataFrame({"Time": [86_399, 1]}))
    assert np.linalg.norm(just_before - just_after) < 1e-3


def test_fitted_preprocessor_survives_pickling(raw_df):
    X = raw_df[RAW_FEATURES]
    pre = build_preprocessor().fit(X)
    restored = pickle.loads(pickle.dumps(pre))
    pd.testing.assert_frame_equal(pre.transform(X), restored.transform(X))
