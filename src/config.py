"""Central configuration for the fraud-detection service.

Every tunable constant and every filesystem path lives here. Nothing else in
the repo should hardcode a path, a threshold, or a connection string -- import
``settings`` instead. Values can be overridden per-environment via ``.env`` or
real environment variables (env vars win).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved from this file's location, so it is correct whether the app is run
# from the repo root, from a subdirectory, or inside a container.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Pydantic v2 reserves the "model_" prefix; we use it for real fields.
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ app
    app_name: str = "fraud-detection-api"
    environment: Literal["local", "docker", "prod"] = "local"
    log_level: str = "INFO"
    random_seed: int = 42

    # ------------------------------------------------------------------ api
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ------------------------------------------- artifact filenames (Phase 1)
    raw_data_filename: str = "creditcard.csv"
    model_filename: str = "isolation_forest.joblib"
    model_metadata_filename: str = "model_metadata.json"

    # ------------------------------------------------- training hyperparams
    test_size: float = Field(0.2, gt=0, lt=1)
    contamination: float = Field(0.0017, gt=0, lt=0.5)  # ~ the Kaggle fraud rate
    n_estimators: int = 200
    max_samples: int | Literal["auto"] = "auto"

    # --------------------------------------------------- decision thresholds
    review_threshold: float = Field(0.50, ge=0, le=1)  # >= this -> manual review
    flag_threshold: float = Field(0.70, ge=0, le=1)  # >= this -> block/flag

    # ----------------------------------------------------- storage (Phase 3)
    database_url: str = "postgresql+psycopg://fraud:fraud@localhost:5432/fraud"

    # --------------------------------------------------- streaming (Phase 4)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "transactions"
    kafka_dlq_topic: str = "transactions.dlq"
    kafka_consumer_group: str = "fraud-scorer"
    producer_rate_per_sec: float = 20.0

    # ------------------------------------------------------- redis (Phase 5)
    redis_url: str = "redis://localhost:6379/0"

    # ----------------------------------------------------------------------
    # Derived paths. Properties, not fields, so they are never read from env.
    # ----------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir / self.raw_data_filename

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def model_metadata_path(self) -> Path:
        return self.models_dir / self.model_metadata_filename

    @property
    def train_split_path(self) -> Path:
        return self.processed_data_dir / "train.parquet"

    @property
    def test_split_path(self) -> Path:
        return self.processed_data_dir / "test.parquet"

    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> Settings:
        if self.flag_threshold < self.review_threshold:
            raise ValueError(
                f"flag_threshold ({self.flag_threshold}) must be >= "
                f"review_threshold ({self.review_threshold})"
            )
        return self

    def ensure_dirs(self) -> None:
        """Create the artifact directories if they do not exist yet."""
        for path in (self.raw_data_dir, self.processed_data_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor -- the .env file is parsed exactly once per process."""
    return Settings()


settings = get_settings()
