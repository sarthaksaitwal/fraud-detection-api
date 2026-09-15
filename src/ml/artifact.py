"""Save and load the trained fraud model (Step 1.8).

The model is written as two files side by side:

    models/fraud_model.joblib     the fitted Pipeline (preprocessing + XGBoost)
    models/model_metadata.json    human-readable facts about that exact file

The metadata records the SHA-256 of the model file, so loading refuses a model
and metadata that don't belong together. It also records the library versions
used in training: a pickled model loaded under different versions can fail or
score differently, so a mismatch raises a warning.

Only load model files you created yourself. Unpickling runs code from the file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from src.config import settings
from src.ml.preprocess import RAW_FEATURES

TRACKED_LIBRARIES = {"scikit-learn": sklearn, "xgboost": xgboost, "pandas": pd, "numpy": np}
RESERVED_KEYS = {
    "model_version",
    "created_at",
    "model_file",
    "model_sha256",
    "input_features",
    "model_features",
    "library_versions",
}


def library_versions() -> dict[str, str]:
    versions = {name: module.__version__ for name, module in TRACKED_LIBRARIES.items()}
    return {"python": platform.python_version(), **versions}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Turn numpy values into plain JSON values, and NaN/infinity into null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_model(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Write the pipeline and its metadata. Returns the metadata as written."""
    clash = RESERVED_KEYS & metadata.keys()
    if clash:
        raise ValueError(f"metadata keys are set by save_model itself: {sorted(clash)}")

    model_path = Path(model_path or settings.model_path)
    metadata_path = Path(metadata_path or settings.model_metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    sha256 = file_sha256(model_path)
    created = datetime.now(timezone.utc)
    full = _json_safe(
        {
            "model_version": f"{created:%Y%m%d-%H%M%S}-{sha256[:8]}",
            "created_at": created.isoformat(timespec="seconds"),
            "model_file": model_path.name,
            "model_sha256": sha256,
            "input_features": list(RAW_FEATURES),
            "model_features": list(pipeline.named_steps["preprocess"].get_feature_names_out()),
            "library_versions": library_versions(),
            **metadata,
        }
    )
    metadata_path.write_text(json.dumps(full, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return full


def load_model(
    model_path: Path | None = None, metadata_path: Path | None = None
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the pipeline after checking it matches its metadata and this code."""
    model_path = Path(model_path or settings.model_path)
    metadata_path = Path(metadata_path or settings.model_metadata_path)
    for path in (model_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; train and save a model first")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Check the fingerprint *before* unpickling, so a mismatched file never runs.
    if file_sha256(model_path) != metadata.get("model_sha256"):
        raise ValueError(
            f"{model_path.name} does not match {metadata_path.name} (SHA-256 differs); "
            "retrain, or restore the matching pair of files"
        )
    if metadata.get("input_features") != list(RAW_FEATURES):
        raise ValueError("the model was trained on different input features than this code sends")

    trained_with = metadata.get("library_versions", {})
    installed = library_versions()
    mismatched = {
        name: {"trained": trained_with.get(name), "installed": installed[name]}
        for name in TRACKED_LIBRARIES
        if trained_with.get(name) != installed[name]
    }
    if mismatched:
        warnings.warn(
            f"model was trained with different library versions: {mismatched}", stacklevel=2
        )

    return joblib.load(model_path), metadata
