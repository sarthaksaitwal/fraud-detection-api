"""Step 1.9 guard rails: the one-command training pipeline."""

import json

import numpy as np
import pytest

from src.ml import train as train_module
from src.ml.artifact import load_model
from src.ml.model import fraud_probability
from src.ml.preprocess import RAW_FEATURES


@pytest.fixture
def trained(raw_df, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("save_splits must not run when save_split_files=False")

    # Also guarantees the test never overwrites the real data/processed files.
    monkeypatch.setattr(train_module, "save_splits", refuse)
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    saved = train_module.train(
        raw=raw_df,
        model_path=model_path,
        metadata_path=metadata_path,
        n_splits=3,
        bootstrap_resamples=20,
        save_split_files=False,
    )
    return saved, model_path, metadata_path


def test_writes_a_model_that_loads_and_scores(trained, raw_df):
    _, model_path, metadata_path = trained
    pipeline, _ = load_model(model_path, metadata_path)
    proba = fraud_probability(pipeline, raw_df[RAW_FEATURES])
    assert ((proba >= 0) & (proba <= 1)).all()


def test_saved_metadata_matches_what_train_returns(trained):
    saved, _, metadata_path = trained
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == saved


def test_metadata_records_thresholds_and_metrics(trained):
    saved, *_ = trained
    thresholds = saved["decision_thresholds"]
    assert 0 < thresholds["review"] < 1
    assert thresholds["block"] is None or thresholds["block"] >= thresholds["review"]
    assert "3-fold" in thresholds["chosen_by"]
    for section in ("out_of_fold_metrics", "test_metrics"):
        assert 0 <= saved[section]["pr_auc"] <= 1
        assert "saving_vs_no_model" in saved[section]["at_decision_thresholds"]
    assert saved["training"]["raw_rows"] == 2000
    assert saved["training"]["duplicates_removed"] == 0


def test_thresholds_come_from_the_cost_model(trained, raw_df):
    saved, *_ = trained
    assert saved["decision_thresholds"]["review"] in np.array(train_module.PROBABILITY_THRESHOLDS)


def test_cli_passes_options_to_train(monkeypatch):
    received = {}
    monkeypatch.setattr(train_module, "train", lambda **kwargs: received.update(kwargs))
    train_module.main(["--n-splits", "4", "--bootstrap-resamples", "50"])
    assert received == {"n_splits": 4, "bootstrap_resamples": 50}
