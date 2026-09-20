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
        # BLOCK_THRESHOLD=none in .env means "never block automatically".
        env_parse_none_str="none",
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
    model_filename: str = "fraud_model.joblib"
    model_metadata_filename: str = "model_metadata.json"

    # ------------------------------------------------- training hyperparams
    test_size: float = Field(0.2, gt=0, lt=1)
    contamination: float = Field(0.0017, gt=0, lt=0.5)  # ~ the Kaggle fraud rate
    n_estimators: int = 200
    max_samples: int | Literal["auto"] = "auto"

    # --------------------------------------------------- decision thresholds
    # Risk is the XGBoost model's predicted fraud probability. Thresholds were
    # chosen in notebooks/008_xgboost_thresholds.ipynb by minimising expected cost
    # on out-of-fold probabilities under the assumptions below.
    review_threshold: float = Field(0.24, ge=0, le=1)  # >= this -> analyst review
    block_threshold: float | None = Field(0.95, ge=0, le=1)  # >= this -> block; None = never

    # ------------------------------- cost assumptions used to choose thresholds
    review_cost: float = Field(5.0, ge=0)  # $ of analyst time per reviewed transaction
    false_block_cost: float = Field(50.0, ge=0)  # $ per legitimate customer blocked
    chargeback_fee: float = Field(15.0, ge=0)  # $ on top of the amount for missed fraud
    max_review_rate: float = Field(0.02, gt=0, le=1)  # analyst capacity, share of traffic

    # ----------------------------------------------------- storage (Phase 3)
    database_url: str = "postgresql+psycopg://fraud:fraud@127.0.0.1:5433/fraud"
    # Record every decision in the database. false: the API scores without one.
    persist_decisions: bool = True

    # --------------------------------------------------- streaming (Phase 4)
    # 127.0.0.1, not localhost: Compose listens on IPv4 only (see DATABASE_URL).
    kafka_bootstrap_servers: str = "127.0.0.1:9092"
    kafka_topic: str = "transactions"
    kafka_dlq_topic: str = "transactions.dlq"
    kafka_consumer_group: str = "fraud-scorer"
    producer_rate_per_sec: float = 20.0
    # Messages scored per model call and saved per database write by the consumer.
    consumer_max_batch: int = Field(500, ge=1, le=10_000)

    # ------------------------------------------------------- redis (Phase 5)
    # 127.0.0.1, not localhost, for the same IPv6 reason as DATABASE_URL.
    redis_url: str = "redis://127.0.0.1:6379/0"
    # Compute velocity features. false: score on the transaction alone.
    velocity_features: bool = True
    # Redis is read and written in front of every score, so it must fail fast
    # rather than hold the request. A local Redis answers in well under a
    # millisecond; anything near this limit means it is in trouble.
    redis_timeout_seconds: float = Field(0.25, gt=0)

    # ------------------------------------------- velocity rules (Step 5.5)
    # Above any of these, an approved transaction is sent for review instead.
    # Chosen from behaviour, never from the fraud labels: the card ids are
    # synthetic, so fitting them to the labels would fit noise. They are then
    # checked against MAX_REVIEW_RATE -- at the test set's own pace they add
    # about 0.8% of traffic to the review queue, inside the 2% analyst budget.
    velocity_max_per_minute: int = Field(5, ge=1)
    velocity_max_per_hour: int = Field(10, ge=1)
    velocity_max_countries: int = Field(2, ge=1)
    velocity_min_gap_seconds: float = Field(2.0, ge=0)

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
        if self.block_threshold is not None and self.block_threshold < self.review_threshold:
            raise ValueError(
                f"block_threshold ({self.block_threshold}) must be >= "
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
