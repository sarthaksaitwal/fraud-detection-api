"""Step 1.8 guard rails: saving and loading the model."""

import json
import subprocess
import sys

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.ml.artifact import file_sha256, load_model, save_model
from src.ml.model import fit_xgboost, fraud_probability
from src.ml.preprocess import RAW_FEATURES, split

FRESH_PROCESS_SCRIPT = """
import json
import sys

import pandas as pd

from src.ml.artifact import load_model

pipeline, metadata = load_model(sys.argv[1], sys.argv[2])
row = pd.DataFrame([json.loads(sys.argv[3])], columns=metadata["input_features"])
print(pipeline.predict_proba(row)[0, 1])
"""


@pytest.fixture
def saved(raw_df, tmp_path):
    X_train, X_test, y_train, _ = split(raw_df)
    pipeline = fit_xgboost(X_train, y_train)
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    save_model(
        pipeline, {"model_type": "xgboost", "note": np.float64(0.5)}, model_path, metadata_path
    )
    return model_path, metadata_path, pipeline, X_test


def _rewrite_metadata(metadata_path, change):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    change(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_round_trip_scores_identically(saved):
    model_path, metadata_path, pipeline, X_test = saved
    loaded, _ = load_model(model_path, metadata_path)
    np.testing.assert_array_equal(
        fraud_probability(loaded, X_test), fraud_probability(pipeline, X_test)
    )


def test_metadata_describes_the_saved_file(saved):
    model_path, metadata_path, *_ = saved
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_sha256"] == file_sha256(model_path)
    assert metadata["model_version"].endswith(metadata["model_sha256"][:8])
    assert metadata["input_features"] == RAW_FEATURES
    assert len(metadata["model_features"]) == 31
    assert metadata["library_versions"]["xgboost"]
    assert metadata["model_type"] == "xgboost"
    assert metadata["note"] == 0.5


def test_a_fresh_python_process_can_load_and_score(saved):
    model_path, metadata_path, pipeline, X_test = saved
    row = X_test.iloc[[0]]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            FRESH_PROCESS_SCRIPT,
            str(model_path),
            str(metadata_path),
            json.dumps(row.iloc[0].to_dict()),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert float(result.stdout.strip()) == pytest.approx(float(fraud_probability(pipeline, row)[0]))


def test_tampered_model_file_is_refused(saved):
    model_path, metadata_path, *_ = saved
    with model_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(model_path, metadata_path)


def test_feature_mismatch_is_refused(saved):
    model_path, metadata_path, *_ = saved
    _rewrite_metadata(metadata_path, lambda m: m.update(input_features=m["input_features"][::-1]))
    with pytest.raises(ValueError, match="features"):
        load_model(model_path, metadata_path)


def test_library_version_mismatch_warns(saved):
    model_path, metadata_path, *_ = saved
    _rewrite_metadata(metadata_path, lambda m: m["library_versions"].update(xgboost="0.0.0"))
    with pytest.warns(UserWarning, match="xgboost"):
        load_model(model_path, metadata_path)


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib", tmp_path / "missing.json")


def test_reserved_metadata_keys_are_rejected(saved, tmp_path):
    _, _, pipeline, _ = saved
    with pytest.raises(ValueError):
        save_model(pipeline, {"model_sha256": "fake"}, tmp_path / "m.joblib", tmp_path / "m.json")


def test_non_finite_numbers_are_written_as_null(saved, tmp_path):
    _, _, pipeline, _ = saved
    save_model(
        pipeline,
        {"alerts_per_fraud": float("inf"), "count": np.int64(3)},
        tmp_path / "m.joblib",
        tmp_path / "m.json",
    )
    metadata = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert metadata["alerts_per_fraud"] is None
    assert metadata["count"] == 3
