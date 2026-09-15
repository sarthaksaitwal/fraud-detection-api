"""Phase 0 guard rails: configuration loads and derived paths are coherent."""

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, Settings, get_settings, settings


def test_settings_load():
    assert settings.app_name == "fraud-detection-api"
    assert settings.environment in {"local", "docker", "prod"}


def test_derived_paths_live_under_project_root():
    for path in (
        settings.raw_data_path,
        settings.processed_data_dir,
        settings.model_path,
        settings.model_metadata_path,
    ):
        assert PROJECT_ROOT in path.parents


def test_model_artifact_name():
    assert settings.model_path.name.endswith(".joblib")
    assert settings.raw_data_path.name == "creditcard.csv"


def test_block_threshold_must_not_undercut_review_threshold():
    with pytest.raises(ValidationError):
        Settings(review_threshold=0.8, block_threshold=0.2)


def test_thresholds_are_bounded():
    with pytest.raises(ValidationError):
        Settings(block_threshold=1.5)


def test_block_threshold_can_be_disabled_from_the_environment(monkeypatch):
    monkeypatch.setenv("BLOCK_THRESHOLD", "none")
    assert Settings().block_threshold is None


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_ensure_dirs_is_idempotent(tmp_path, monkeypatch):
    settings.ensure_dirs()
    settings.ensure_dirs()
    assert settings.raw_data_dir.is_dir()
    assert settings.models_dir.is_dir()
